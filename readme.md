# RNN Architectures

Repo of RNN cell architectures from various papers, including their implementation,
testing, and studies of them in various model architectures.

## Using and viewing

Implementations are done in [PyTorch](https://pytorch.org/) (v0.3 for now). All
model implementations are done as extending the `nn.Module` class, and in a way
that lets them be easily imported and used, following the documented input and
output sizes.

Tests and studies are done in Jupyter notebooks which are viewable (but not
runnable) without having PyTorch installed.

## References

* [Nested LSTMs](https://arxiv.org/abs/1801.10308) (Moniz, Krueger. 2018): LSTM variant where the cell state is calculated using a stateful function; the paper uses a classical LSTM as this function, but this implementation allows for any type of function to be used.
