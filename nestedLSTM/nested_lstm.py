"""
Implements a Nested LSTM (nLSTM), as described in the paper "Nested LSTMs"
(Moniz, Krueger. 2017).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter
from torch.nn.modules.rnn import RNNBase, RNNCellBase

class NestedLSTM(RNNBase):
    def __init__(self, input_size, hidden_size, state_fn,
                 num_layers = 1, bias = True, batch_first = False,
                 dropout = 0, bidirectional = False, identity_fn = False):
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

        
        Note that if `state_fn` is passed in as a callable implementing simple
        addition, e.g. something like:  
            lambda x_i, (h_i, c_i, s_i): x_i + h_i  
        then this essentially reduces to a classical LSTMCell implementation.
        Also note that this input can't be a lambda function if you want this
        class to be picklable.  
        '''
        super(NestedLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.state_fn = state_fn
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_state = {}
        self.bidirectional = bidirectional
        self.identity_fn = identity_fn
        

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
        self.state_fn = state_fn
        self.identity_fn = identity_fn

        # Initialize weights and biases. Sizes same as an LSTM cell
        self.weight_ih = Parameter(torch.Tensor(4*hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4*hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4*hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4*hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdev = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdev, stdev)

    def forward(self, input, hx):
        '''
        Inputs:   
        `input`: of size `(seq, batch, features)`  
        `hx`: tuple of `(h_0, c_0, s_0)`. `c_0` and `s_0` can be `None` if the
        inner cell isn't stateful. If `s_0` (any needed states for the internal
        cell) is an iterable (list or tuple), the first item will be extracted
        while the rest are passed as an iterable.   

        Outputs:  
        `h_1, c_1, s_1`
        '''
        h_0, c_0, s_0 = hx

        # Apply affine ransforms on input and h_0
        x_t = F.linear(input, self.weight_ih, bias = self.bias_ih)
        h_t = F.linear(  h_0, self.weight_hh, bias = self.bias_hh)
        affine = x_t + h_t
        # Pass through the respective activation functions and separate
        fio_1 = F.sigmoid(affine[:, :3*self.hidden_size])
        g_1   = F.tanh(affine[:, 3*self.hidden_size:])

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


