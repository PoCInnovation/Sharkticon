import time
import matplotlib.pyplot as plts
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchvision import datasets, transforms
import torch.nn.functional as F

#On transforme les images en tensor, et on les normalise pour les traiter beaucoup plus facilement.
#On choisit un batch_size de 32 c'est à dire qu'on travaille par paquets de 32 images.
#pour les weight, on initialise 784 weight * 10 dimensions.
#Les images font 28*28 pixels soit 784 pixel par images. de plus on a 10 classes possible (0 à 9) donc on a 10 dimensions.
#le but est d'avoir une propabilité sur chaque classe.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
batch_size = 32
train_loader = torch.utils.data.DataLoader(datasets.MNIST('./', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./', train=False, download=True, transform=transform), batch_size=batch_size, shuffle=True)
weights = torch.randn(784, 10, requires_grad=True)
nb_iteration = 0;
nb_iteration_max = 1000 ##Durée de l'entrainement

def test(weights, test_loader):
    test_size = len(test_loader.dataset)
    correct = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.view((-1, 28*28))
        outputs = torch.matmul(data, weights) #Permet de multiplier deux tensor, weight et data.
        softmax = F.softmax(outputs, dim=1) #Le softmax permet d'obtenir la probabilité.
        pred = softmax.argmax(dim=1, keepdim=True)
        n_correct = pred.eq(target.view_as(pred)).sum().item() #On verifie si la prédiction est juste.
        correct += n_correct
    acc = correct / test_size
    print("Precision : ", acc)
    return

def train():
    for batch_idx, (data, targets) in enumerate(train_loader):
        if weights.grad is not None: ##Lors de l'entrainement, on reset les poids pour chaque image.
            weights.grad.zero_()

        data = data.view((-1, 28*28))
        outputs = torch.matmul(data, weights)
        log_softmax = F.log_softmax(outputs, dim=1)
        loss = F.nll_loss(log_softmax, targets)
        loss.backward()

        with torch.no_grad(): #On effectue la descente de gradient.
            weights -= 0.1*weights.grad

        nb_iteration += 1
        if nb_iteration % 100 == 0:
            test(weights, test_loader)

        if nb_iteration > nb_iteration_max: #On arrête l'entrainement au bout de n itérations.
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