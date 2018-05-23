import torch
import torch.nn as nn

from nested_lstm import NestedLSTM, NestedLSTMCell

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
