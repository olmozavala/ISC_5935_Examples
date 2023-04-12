import torch
import torch.nn as nn

input_size = 1
hidden_size = 2
num_layers = 2
batch_size = 1

rnn = nn.RNN(input_size, hidden_size, num_layers)
# rnn = nn.GRU(input_size, hidden_size, num_layers)
# rnn = nn.LSTM(input_size, hidden_size, num_layers)

print(f"Input size: {input_size}, Hidden size: {hidden_size}, Number of layers: {num_layers}")
total_weights = sum(p.numel() for p in rnn.parameters() if p.requires_grad)
print(f"Number of weights in the model: {total_weights}")
# Print tha values of the weights
for name, param in rnn.named_parameters():
    if name.find('weight') != -1:
        # param.data.fill_(1)
        x = 1 # Do nothing
    else:
        param.data.fill_(0)
    print(f"Name: {name} \t Size: {param.size()} \t Values: {param[:].data} \n")

x = torch.ones(batch_size, input_size)*2
h = torch.ones(num_layers, hidden_size)*0

output, hn = rnn(x, h)
print(f"Input x_t: {x.data}, value h_t-1: {h.data}")
print(f"Output size: {output.size()}, value O_t: {output.data}")
print(f"Hidden state size: {hn.size()}, value h_t: {hn.data}")

#%%

