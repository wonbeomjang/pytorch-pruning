import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from tqdm import tqdm
import numpy as np
from time import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Preparing data..')

transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

def train(num_epoch, net, criterion, optimizer, dataloader):
  net = net.train()
  for epoch in range(num_epoch):
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for batch_idx, (inputs, targets) in pbar:
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = net(inputs)
      loss = criterion(outputs, targets)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      with torch.no_grad():
        train_loss += loss.data
        outputs = outputs.argmax(dim=1)
        correct += (targets == outputs).sum()
        total += inputs.size(0)
      pbar.set_description(f"[{epoch}/{num_epoch}] Loss: {train_loss / total:.4f}, Accuracy: {correct / total:.4f}")

def test(net, criterion, dataloader):
  net.eval()
  test_loss = 0
  correct = 0
  num_data = 0
  cur = time()


  with torch.no_grad():
    pbar = tqdm(dataloader, total=len(dataloader))
    for data, target in pbar:
      data, target = data.to(device), target.to(device)
      output = net(data)
      test_loss += criterion(output, target).data
      output = output.argmax(dim=1)
      correct += (target == output).sum()
      num_data += data.size(0)
    
      pbar.set_description(f"Test set: Average loss: {test_loss / num_data:.4f}, Accuracy: {correct / num_data:.4f}, Time cost: {time() - cur:.4f}")

def pruning(net):
  for name, module in net.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
      prune.ln_structured(module, 'weight', amount=0.3, n=2, dim=0)
      prune.remove(module, 'weight')

net = models.vgg16(pretrained=True)
net.classifier = nn.Sequential(
  nn.Linear(512 * 7 * 7, 4096),
  nn.ReLU(True),
  nn.Dropout(p=0.5),
  nn.Linear(4096, 4096),
  nn.ReLU(True),
  nn.Dropout(p=0.5),
  nn.Linear(4096, 10),
)
net = net.to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
epoch = 1

for i in range(10):
  train(epoch, net, criterion, optimizer, trainloader)
  print("Before Pruning...")
  test(net, criterion, testloader)
  pruning(net)
  print("After Pruning...")
  test(net, criterion, testloader)
