import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from medmnist import BreastMNIST

#torch / nn / optim → PyTorch, para criar e treinar a rede neural.
#transforms → para pré-processar as imagens (resize, tensor, normalização).
#DataLoader → carregar os dados em batches.
#BreastMNIST → dataset de imagens de mama, já dividido em train/test.

# Transformações nas imagens
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((28,28)),
    transforms.Normalize([0.5], [0.5])
])

# Dataset BreastMNIST
train_ds = BreastMNIST(split="train", transform=transform, download=True)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

# Modelo simples
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(16*14*14, 2)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-3)

# Treino rápido (3 épocas só para exemplo)
for epoch in range(3):
    model.train()
    running = 0
    for imgs, labels in train_loader:
        out = model(imgs)
        loss = criterion(out, labels.squeeze().long())
        opt.zero_grad()
        loss.backward()
        opt.step()
        running += loss.item()
    print(f"Época {epoch+1}, Loss: {running/len(train_loader):.4f}")

# Guardar modelo treinado
torch.save(model.state_dict(), "breast_model.pth")
print("Modelo salvo em breast_model.pth")