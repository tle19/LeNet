import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.optim as O

#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#CIFAR-10 & CIFAR-100 dataset
# train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
# val_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=ToTensor())

train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=ToTensor())
val_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=ToTensor())

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1000, shuffle=False)

#model
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

def LeNet(in_channels=3, init_padding=0, num_classes=10,  activation=nn.ReLU):
    net = nn.Sequential(
        nn.Conv2d(in_channels, 6, kernel_size=5, padding=init_padding), activation(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), activation(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        Flatten(),
        nn.Linear(16*5*5, 120), activation(),
        nn.Linear(120, 84), activation(),
        nn.Linear(84, num_classes)
    ) 
    return net

model = LeNet(in_channels=3, num_classes=100)
model = model.to(device)
opt = O.Adam(model.parameters(), lr=0.0009)
criterion = nn.CrossEntropyLoss(reduction="mean")

train_cross_entropy = []
train_accuracy = []
validation_cross_entropy = []
validation_accuracy = []
best_model_accuracy = 0

validate_every = 100
epochs = 50

#training & summary statistics
for epoch in range(epochs):
    n_correct = 0
    n_total = 0
    for i, batch in enumerate(train_loader):
        x, labels = batch
        x, labels = x.to(device), labels.to(device)
        N = x.shape[0]
        
        model.train()
        
        opt.zero_grad()
        
        y_hat = model(x)
        loss = criterion(y_hat, labels)
        loss.backward()
        opt.step()
        
        n_correct += (torch.argmax(y_hat, dim=1) == labels).sum().item()
        n_total += N

        train_accuracy.append(n_correct / n_total)
        train_cross_entropy.append(loss)
        
        # evaluation mode
        model.eval()
        if i % validate_every == 0:
            n_val_correct = 0
            n_val_total = 0
            v_cross_entropy_sum = 0
            
            with torch.no_grad():
                for j, v_batch in enumerate(val_loader):
                    v_x, v_labels = v_batch
                    v_x, v_labels = v_x.to(device), v_labels.to(device)
                    v_N = v_x.shape[0]
                    
                    v_y_hat = model(v_x)
                    v_loss = criterion(v_y_hat, v_labels)
                    v_cross_entropy_sum += v_loss
                    n_val_correct += (torch.argmax(v_y_hat, dim=1) == v_labels).sum().item()
                    n_val_total += v_N

            print(f"EPOCH {epoch + 1}: Batch {i} \t loss: {loss}, accuracy: {n_correct / n_total}, \n\t\t\t val_loss: {v_cross_entropy_sum / n_val_total}, val_accuracy: {n_val_correct / n_val_total}")
            validation_accuracy.append(n_val_correct / n_val_total)
            validation_cross_entropy.append(v_cross_entropy_sum / n_val_total)
            if n_val_correct / n_val_total >= best_model_accuracy:
                best_model_accuracy = n_val_correct / n_val_total
                print("saving")
                torch.save(model.state_dict(), './lenet5_cifar100')