"""
Implements a Nested LSTM (nLSTM), as described in the paper "Nested LSTMs"
(Moniz, Krueger. 2017).
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.modules.rnn import RNNCellBase
from torch.nn.utils.rnn import PackedSequence

class NestedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, state_fn,
                 num_layers = 1, bias = True, batch_first = False,
                 dropout = 0, bidirectional = False, identity_fn = False):
        '''
        At the moment, this class is implemented as pretty literally a list of
        `NestedLSTMCell`s, with the inputs passed to this module when called
        fed forward through the layers one step at a time. Ideally, this would
        be implemented in C with Python hooks to call it, but for development
        and testing purposes, this should do for now.

            Inputs:  
        `input_size`: the number of expected features in the input x  
        `hidden_size`: the number of features in the hidden state  
        `state_fn`: a list of length `num_layers` of callables (torch Module,
        function, lambda function, etc.) implementing the (usually stateful)
        function to use to calculate this cell's cell state each step. `state_fn`
        must take two input arguments: the input `x_i` and a 3-tuple of `(h_i,
        c_i, s_i)`, which are the inner cell's hidden state and cell state (if
        any) and any other states that need to be passed through, respectively.
        This can also be a single callabe, in which case it'll be duplicated and
        used for each layer.  
        `num_layers`: number of recurrent layers  
        `bias`: if `False`, then the layer does not use bias weights `b_ih` and
        `b_hh`. Defaults to `True`  
        `batch_first`: if `True`, then the input and output tensors are provided
        as `(batch, seq, features)`
        `dropout`: if non-zero, introduces a dropout layer on the outputs of
        each RNN layer except the last layer
        `bidirectional`: if `True`, becomes a bidirectional RNN
        `identity_fn`: if `True`, uses the identity function when calculating
        the hidden state of the cell. If `False`, uses a tanh function. I.e.,
        determines the function f() used when calculating:  
            h_t = o_t * f(c_t)   
        This can also be a list of length `num_layers` of booleans so that each
        layer calculates its hidden state differently.
        
        Note that if `state_fn` is passed in as a callable implementing simple
        addition, e.g. something like:  
            lambda x_i, (h_i, c_i, s_i): x_i + h_i  
        then this essentially reduces to a classical LSTM implementation. Also
        note that this input can't be a lambda function if you want this class
        to be picklable.  
        '''
        super(NestedLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_state = {}
        self.bidirectional = bidirectional
        if not isinstance(state_fn, list):
            state_fn = [copy.deepcopy(state_fn) for _ in range(num_layers)]
        assert len(state_fn) == num_layers
        self.state_fn = state_fn
        if not isinstance(identity_fn, list):
            identity_fn = [identity_fn for _ in range(num_layers)]
        assert len(identity_fn) == num_layers
        self.identity_fn = identity_fn
        num_directions = 2 if bidirectional else 1
        # Initialize an nLSTM cell for each direction and each layer
        self.all_cells = nn.ModuleList([])
        for layer, (st_fn, id_fn) in enumerate(zip(state_fn, identity_fn)):
            layer_cells = nn.ModuleList([])
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                cell = NestedLSTMCell(
                    layer_input_size, hidden_size, st_fn, bias, id_fn
                )
                suffix = '_reverse' if direction == 1 else ''
                cell.name = 'cell_l{}{}'.format(layer, suffix)
                layer_cells.append(cell)
            self.all_cells.append(layer_cells)
        self.reset_parameters()
    
    def reset_parameters(self):
        '''
        Initializes using Glorot normal initialization on the input gate weights
        and orthogonal initialization for the rest.
        '''
        for layer in self.all_cells:
            for cell in layer:
                cell.reset_parameters()

    def forward(self, inputs, hx):
        '''
        Inputs:   
        `inputs`: tensor of size `(seq, batch, input_size)` containing the
        features of the input sequence. Currently doesn't allow for
        `PackedSequence` type inputs :/  
        `hx`: tuple containing `(h_0, c_0, s_0)`, where...  
        `h_0`: tensor of size `(num_layers*num_directions, batch, hidden_size)`
        containing the initial hidden states for each element in the batch  
        `c_0`: tensor of size `(num_layers*num_directions, batch, hidden_size)`
        containing the initial cell state for each element in the batch  
        `s_0`: list of length `num_layers` of lists of needed states for each
        direction of each layer; if `bidirectional==False`, then the state lists
        for each layer will be of length 1. Each item in the inner lists can be
        a `None` if that layer is not stateful, a single Tensor state, or an
        iterable (list or tuple) of states (useful if there are multiply nested
        cells in a layer). If one of the items in `s_0` is an interable, the
        first item will be extracted as the state for the first depth of nested
        cells, and the rest will be passed as an iterable to the remaining
        nested cells. Basically, this follows the same input format as the input
        `s_0` into a `NestedLSTMCell`.  

        Outputs:  
        `output`: tensor of size `(seq, batch, hidden_size*num_directions)`
        containing the output features (`h_t`) from the last layer of the
        NestedLSTM stack, for each t.  
        `hx_n`: tuple containing `(h_n, c_n, s_n)`, where...  
        `h_n`: tensor of size `(num_layers*num_directions, batch, hidden_size)`
        containing the hidden state for each layer and direction at the last
        time step (`t=seq_len`).  
        `c_n`: tensor of size `(num_layers*num_directions, batch_hidden_size)`
        containing the cell state for each layer and direction at the last time
        step (`t=seq_len`) for each layer and direction.  
        `s_n`: a list of lists in the same format at `s_0` in the input `hx`
        containing the resulting inner cell states for each layer and direction
        at the last time step (`t=seq_len`).  
        '''
        h_0, c_0, s_0 = hx
        assert len(s_0) == self.num_layers

        seq_dim = 1 if self.batch_first else 0
        seq_len = inputs.size(seq_dim)

        # Split up h_0 and c_0 by the first dimension
        assert h_0.size(0) == c_0.size(0)
        hs = h_0.chunk(num_chunks = self.num_layers, dim = 0)
        cs = c_0.chunk(num_chunks = self.num_layers, dim = 0)

        # Feed forward through the layers
        h_n = []
        c_n = []
        s_n = []
        rnn_in = [
            h_t.squeeze(seq_dim) for h_t in inputs.chunk(
                num_chunks = seq_len, dim = seq_dim
            )
        ]
        for cell, h, c, s in zip(self.all_cells, hs, cs, s_0):
            # Prep any necessary setup for a bidirectional input
            if self.bidirectional:
                # Split the forward and reverse cells
                cell_r = cell[1]
                # Split the forward and reverse states up
                h, h_r = h.chunk(2, dim = 0)
                c, c_r = c.chunk(2, dim = 0)
                h_r = h_r.squeeze(0)
                c_r = c_r.squeeze(0)
                s_r = s[1]
            cell = cell[0]
            h = h.squeeze(0)
            c = c.squeeze(0)
            s = s[0]
            # Run the sequence through this layer's cells one step at a time
            rnn_out = []
            s_layer = []
            for t in range(seq_len):
                rnn_t = rnn_in[t]
                h, c, s = cell(rnn_t, (h, c, s))
                rnn_out.append(h)
            # Save the final states for this layer
            h_n.append(h.unsqueeze(0))
            c_n.append(c.unsqueeze(0))
            s_layer.append(s)
            # Reverse direction
            if self.bidirectional:
                rnn_r_out = []
                for t in range(seq_len-1, -1, -1):
                    rnn_t = rnn_in[t]
                    h_r, c_r, s_r = cell_r(rnn_t, (h_r, c_r, s_r))
                    rnn_r_out.append(h_r)
                h_n.append(h_r.unsqueeze(0))
                c_n.append(c_r.unsqueeze(0))
                s_layer.append(s_r)
                # Concatenate the outputs of this layer in both directions
                rnn_out = [
                    torch.cat([h_t, h_r_t], -1) for h_t, h_r_t in zip(
                        rnn_out, rnn_r_out
                    )
                ]
            s_n.append(s_layer)
            rnn_in = rnn_out
            # End feed forward loop through time into Nested LSTM stack
        # Concatenate each output
        h_n = torch.cat(h_n, 0)
        c_n = torch.cat(c_n, 0)
        output = torch.cat([o_t.unsqueeze(seq_dim) for o_t in rnn_out], seq_dim)
        # Aaaaand return!
        return output, (h_n, c_n, s_n)


class NestedLSTMCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, state_fn,
                 bias = True, identity_fn = False):
        '''
            Inputs:  
        `input_size`: the number of expected features in the input x  
        `hidden_size`: the number of features in the hidden state  
        `state_fn`: a callable (torch Module, function, lambda function, etc.)
        implementing the (usually stateful) function to use to calculate this
        cell's cell state each step. `state_fn` must take two input arguments:
        the input `x_i` and a 3-tuple of `(h_i, c_i, s_i)`, which are the inner
        cell's hidden state and cell state (if any) and any other states that
        need to be passed through, respectively.  
        `bias`: if `False`, then the layer does not use bias weights `b_ih` and
        `b_hh`. Defaults to `True`  
        `identity_fn`: if `True`, uses the identity function when calculating
        the hidden state of the cell. If `False`, uses a tanh function. I.e.,
        determines the function f() used when calculating:  
            h_t = o_t * f(c_t)  
    
        Note that if `state_fn` is passed in as a callable implementing simple
        addition, e.g. something like:  
            lambda x_i, (h_i, c_i, s_i): x_i + h_i  
        then this essentially reduces to a classical LSTMCell implementation.
        Also note that this input can't be a lambda function if you want this
        class to be picklable.  
        '''
        super(NestedLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.identity_fn = identity_fn

        # Initialize weights and biases. Sizes same as an LSTM cell
        # Weights are: [forget, input, output, input-transform]
        self.weight_ih = Parameter(torch.Tensor(4*hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4*hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4*hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4*hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()
        self.state_fn = state_fn

    def reset_parameters(self):
        '''
        Initializes using Glorot normal initialization on the input gate weights
        and orthogonal initialization for the rest.
        '''
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.orthogonal(p)
            else:
                nn.init.constant(p, 0)
        # Input transform gate get Xavier (Glorot) normal initialization
        nn.init.xavier_normal(self.weight_ih[3*self.hidden_size:, :])
        nn.init.xavier_normal(self.weight_hh[3*self.hidden_size:, :])

    def forward(self, inputs, hx):
        '''
        Inputs:   
        `inputs`: of size `(batch, input_size)`  
        `hx`: tuple of `(h_0, c_0, s_0)`. `s_0` can be `None` if the inner cell
        isn't stateful. If `s_0` (any needed states for the internal cell) is an
        iterable (list or tuple), the first item will be extracted while the
        rest are passed as an iterable.   

        Outputs:  
        `h_1, c_1, s_1`
        '''
        h_0, c_0, s_0 = hx

        # Apply affine ransforms on input and h_0
        x_t = F.linear(inputs, self.weight_ih, bias = self.bias_ih)
        h_t = F.linear(   h_0, self.weight_hh, bias = self.bias_hh)
        affine = x_t + h_t
        # Pass through the respective activation functions and separate
        fio_1 = F.sigmoid(affine[:, :3*self.hidden_size])
        g_1   = F.tanh(   affine[:, 3*self.hidden_size:])

        f_1 = fio_1[:,                    :   self.hidden_size]
        i_1 = fio_1[:,   self.hidden_size : 2*self.hidden_size]
        o_1 = fio_1[:, 2*self.hidden_size : ]

        # Create the inputs to the inner cell function
        x_i = g_1.mul(i_1)
        h_i = c_0.mul(f_1)
        if type(s_0) in [tuple, list]:
            c_i = s_0[0]
            s_0 = s_0[1:]
            if len(s_0) == 0:
                s_0 = None
            elif len(s_0) == 1:
                s_0 = s_0[0]
        else:
            c_i = s_0
            s_0 = None

        # Get the outer cell state
        i_pkg = (h_i, c_i) if s_0 is None else (h_i, c_i, s_0)
        c_1 = self.state_fn(x_i, i_pkg)
        if type(c_1) in [tuple, list]:
            s_1 = c_1[1:]
            c_1 = c_1[0]
            if len(s_1) == 0:
                s_1 = None
            elif len(s_1) == 1:
                s_1 = s_1[0]
        else:
            s_1 = None

        # Get the outer hidden state
        if not self.identity_fn:
            c_1 = F.tanh(c_1)
        h_1 = c_1.mul(o_1)

        return h_1, c_1, s_1


