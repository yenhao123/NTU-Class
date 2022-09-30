import torch
import torch.nn as nn

rnn = nn.RNN(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
print(input)
output, hn = rnn(input, h0)
