import torch
import torch.nn as nn
import numpy as np

input_size = 1
hidden_size = 1
num_layers = 1
batch_size = 1

rnn = nn.RNN(input_size, hidden_size, num_layers)
# rnn = nn.GRU(input_size, hidden_size, num_layers)
# rnn = nn.LSTM(input_size, hidden_size, num_layers)

print(f"Input size: {input_size}, Hidden size: {hidden_size}, Number of layers: {num_layers}")
total_weights = sum(p.numel() for p in rnn.parameters() if p.requires_grad)
print(f"Number of weights in the model: {total_weights}")
# Print tha values of the weights
#%% Manually compute (DELETE)
x = 0.5
h = 0
wi, wh, bi, bh =  (0.5, -0.3, -0.2, 0.8)
h = np.tanh(x*wi + bi + h*wh + bh)
print(f"Output: {h}")
h = np.tanh(x*wi + bi + h*wh + bh)
print(f"Output: {h}")

#%%
x = torch.ones(1, 1)*x
h = torch.zeros(1, 1)*h
rnn.bias_hh_l0.data.fill_(bh)
rnn.bias_ih_l0.data.fill_(bi)
rnn.weight_hh_l0.data.fill_(wh)
rnn.weight_ih_l0.data.fill_(wi)
output, h = rnn(x, h)
print(f"Output {output.data}")
output, h = rnn(x, h)
print(f"Output {output.data}")

#%%

# for name, param in rnn.named_parameters():
#     if name.find('weight') != -1:
#         param.data.fill_(1)
#         x = 1 # Do nothing
#     else:
#         param.data.fill_(0)
#     print(f"Name: {name} \t Size: {param.size()} \t Values: {param[:].data} \n")

x = torch.ones(batch_size, input_size)*1
x2 = torch.ones(batch_size, input_size)*1
h = torch.ones(num_layers, hidden_size)*0

# Showing that it is the same to call the rnn twice than to call it once with two inputs
print("------------- Calling the RNN first time -------------")
output, hn = rnn(x, h)
print(f"Input x_t: {x.data}, value h_t-1: {h.data}")
print(f"Output size: {output.size()}, value O_t: {output.data}")
print(f"Hidden state size: {hn.size()}, value h_t: {hn.data}")
print("------------- Calling the RNN second time -------------")
output, hn = rnn(x, hn)
print(f"Input x_t: {x2.data}, value h_t-1: {hn.data}")
print(f"Output size: {output.size()}, value O_t: {output.data}")
print(f"Hidden state size: {hn.size()}, value h_t: {hn.data}")
#%%
print("------------- Calling the RNN with two inputs --------------")
x3 = torch.cat((x, x2), dim=0)
output, hn = rnn(x3, h)
print(f"Input x_t: {x3.data}, value h_t-1: {h.data}")
print(f"Output size: {output.size()}, value O_t: {output.data}")
print(f"Hidden state size: {hn.size()}, value h_t: {hn.data}")

#%% BiRNNs
input_size = 16  # embedding size
hidden_size = 8
num_layers = 2
batch_size = 1
sequence_size = 4  # number of words in a sentence
bidirectional = False

rnn = nn.RNN(input_size, hidden_size, num_layers, bidirectional=bidirectional)

print(f"Input size: {input_size}, Hidden size: {hidden_size}, Number of layers: {num_layers}")
total_weights = sum(p.numel() for p in rnn.parameters() if p.requires_grad)
print(f"Number of weights in the model: {total_weights}")

if bidirectional:
    sequence_size = sequence_size * 2

x = torch.ones(batch_size, sequence_size, input_size)
h = torch.ones(num_layers, sequence_size, hidden_size)

output, hn = rnn(x, h)
print(f"Input x_t: {x.data.size()}, value h_t-1: {h.data.size()}")
print(f"Output size: {output.size()}")
print(f"Hidden state size: {hn.size()}")
#%%

