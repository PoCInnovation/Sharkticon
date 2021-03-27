import time
import matplotlib.pyplot as plts
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchvision import datasets, transforms
import torch.nn.functional as F

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
batch_size = 32
train_loader = torch.utils.data.DataLoader(datasets.MNIST('./', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./', train=False, download=True, transform=transform), batch_size=batch_size, shuffle=True)
weights = torch.randn(784, 10, requires_grad=True)
nb_iteration = 0;
nb_iteration_max = 1000

def test(weights, test_loader):
    test_size = len(test_loader.dataset)
    correct = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.view((-1, 28*28))
        outputs = torch.matmul(data, weights)
        softmax = F.softmax(outputs, dim=1)
        pred = softmax.argmax(dim=1, keepdim=True)
        n_correct = pred.eq(target.view_as(pred)).sum().item()
        correct += n_correct
    acc = correct / test_size
    print("Precision : ", acc)
    return

def train():
    for batch_idx, (data, targets) in enumerate(train_loader):
        if weights.grad is not None:
            weights.grad.zero_()

        data = data.view((-1, 28*28))
        outputs = torch.matmul(data, weights)
        log_softmax = F.log_softmax(outputs, dim=1)
        loss = F.nll_loss(log_softmax, targets)
        loss.backward()

        with torch.no_grad():
            weights -= 0.1*weights.grad

        nb_iteration += 1
        if nb_iteration % 100 == 0:
            test(weights, test_loader)

        if nb_iteration > nb_iteration_max:
            break


train()
batch_idx, (data, target) = next(enumerate(test_loader))
data = data.view((-1, 28*28))

outputs = torch.matmul(data, weights)
softmax = F.softmax(outputs, dim=1)
pred = softmax.argmax(dim=1, keepdim=True)

plt.imshow(data[0].view(28, 28), cmap="gray")
plt.title("Predicted class {}".format(pred[0]))
plt.show()