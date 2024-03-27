# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D

# Define the vocabulary size and the embedding dimension
vocab_size = 5
embedding_dim = 3

# Create the embedding layer
embedding_layer = nn.Embedding(vocab_size, embedding_dim)
# Example indices of words in the vocabulary
word_indices = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
# Get the embeddings for the word indices
word_embeddings = embedding_layer(word_indices)
# Print the embeddings
print(word_embeddings)

#%% A function to plot the embeddings in 3D (showing relationships)
def plot_embeddings(embeddings, words):
    # Plot the 3D embeddings
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, word in enumerate(words):
        x, y, z = word_embeddings[i].detach().numpy()
        ax.scatter(x, y, z, label=word)
        ax.text(x, y, z, word)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()

words = ['dog', 'cat', 'horse', 'house', 'car']
plot_embeddings(word_embeddings, words)

# %%
# Initialize training optimizer and random target embeddings
print(word_embeddings)
# target_embeddings = torch.randn_like(word_embeddings)
# Make a tensor of size (5, 3) with all values as 1
target_embeddings = torch.tensor([[1.0, 0.0, 0.0],
                                  [0.9, 0.0, 0.0],
                                  [1.1, 0.0, 0.0],
                                  [0.0, 1.0, 1.0],
                                  [0.0, 1.1, 1.0],
                                  ])
print(target_embeddings)
# %%
# Define the loss function (Mean Squared Error in this case)
loss_function = nn.MSELoss()
optimizer = optim.SGD(embedding_layer.parameters(), lr=0.1)

#%%
# Train the embedding layer
epochs = 200
for i in range(100):
    optimizer.zero_grad()
    word_embeddings = embedding_layer(word_indices)
    loss = loss_function(word_embeddings, target_embeddings)
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f"Epoch: {i}, Loss: {loss.item()}")
        print(word_embeddings)
        plot_embeddings(word_embeddings, words)
#%%

