#https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627
#https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e

import numpy as np
import torch
import torchvision as TV
import matplotlib.pyplot as plt
from time import time
from torch import nn
from torch import optim

# ? GET_DATA

transform = TV.transforms.Compose([TV.transforms.ToTensor(),
                              TV.transforms.Normalize((0.5,), (0.5,)),
                              ])

trainset = TV.datasets.MNIST('./files/MNIST/', download=True, train=True, transform=transform)
valset = TV.datasets.MNIST('./files/MNIST/', download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)


def print_data(trainloader):
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(type(images))
    print(images.shape)
    print(labels.shape)
    plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
    igure = plt.figure()
    num_of_images = 60
    for index in range(1, num_of_images + 1):
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')

# ? Define Neural Network

input_size = 784
hidden_sizes = [128, 64]
output_size = 10
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(), # ! ReLU activation ( a simple function which allows positive values to pass through, whereas negative values are modified to zero )
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size),
            nn.LogSoftmax(dim=1)) # ! activate if classification pb else 0

print(model)

criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images)
loss = criterion(logps, labels)

print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)

def Test_train():
    #Gradient Descent optimization
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    print('Initial weights - ', model[0].weight)
    images, labels = next(iter(trainloader))
    images.resize_(64, 784)
    # Clear the gradients, do this because gradients are accumulated
    optimizer.zero_grad()
    # Forward pass, then backward pass, then update weights
    output = model(images)
    loss = criterion(output, labels)
    loss.backward()
    print('Gradient -', model[0].weight.grad)
    optimizer.step()
    print('Updated weights - ', model[0].weight)

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 3
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1) # Flatten MNIST images into a 784 long vector
        optimizer.zero_grad() # Training pass
        output = model(images)
        loss = criterion(output, labels)
        loss.backward() #This is where the model learns by backpropagating
        optimizer.step() # And optimizes its weights here
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
print("\nTraining Time (in minutes) =",(time()-time0)/60)

def view_classify(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()


images, labels = next(iter(valloader))
img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model(img)
# Output of the network are log-probabilities, need to take exponential for probabilities
ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))
view_classify(img.view(1, 28, 28), ps)

correct_count, all_count = 0, 0
for images,labels in valloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    # Turn off gradients to speed up this part
    with torch.no_grad():
        logps = model(img)

    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))