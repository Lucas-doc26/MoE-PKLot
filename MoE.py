import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ======================
# Dataset
# ======================
class ImageDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

        self.transform = transforms.Compose([
            transforms.Resize((124, 124)),   # garante 124x124
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], #padrão ImageNet
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]

        image = Image.open(img_path).convert('RGB')
        image = np.array(image).transpose(2, 0, 1)  # [H,W,C] -> [C,H,W]
        image = torch.from_numpy(image).float() / 255.0

        image = self.transform(image)

        label = int(self.data.iloc[idx, 1])

        return image, label



# Expert CNN
class ExpertCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),     # 124 -> 62
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )

        self.fc = nn.Linear(64 * 62 * 62, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)



# Router (Gating Network)
class Router(nn.Module):
    def __init__(self, n_experts, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(3, n_experts)

    def forward(self, x):
        # x: [B, C, H, W]
        x_pooled = self.pool(x).view(x.size(0), -1)  # [B, 3]
        logits = self.linear(x_pooled)  # [B, n_experts]
        
        # seleciona os top_k melhores experts por amostra
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=1)  # [B, top_k]
        
        # Normaliza apenas os top-k
        top_k_gates = F.softmax(top_k_logits, dim=1)  # [B, top_k]
        
        # Cria matriz de gates completa (zeros para não-selecionados)
        gates = torch.zeros_like(logits)  # [B, n_experts]
        gates.scatter_(1, top_k_indices, top_k_gates)
        
        return gates, top_k_indices



# ======================
# MoE
# ======================
class MoECNN(nn.Module):
    def __init__(self, n_experts, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([ExpertCNN() for _ in range(n_experts)])
        self.router = Router(n_experts, top_k=top_k)

    def forward(self, x):
        # Router seleciona top-k experts
        gates, top_k_indices = self.router(x)  # gates: [B, n_experts], top_k_indices: [B, top_k]
        
        # Executa apenas os experts selecionados (mais eficiente)
        batch_size = x.size(0)
        expert_outputs = []
        
        for i in range(self.n_experts):
            expert_outputs.append(self.experts[i](x))
        
        outputs = torch.stack(expert_outputs, dim=1)  # [B, n_experts, C]
        
        # Aplica gates (ponderação dos experts)
        gates = gates.unsqueeze(-1)  # [B, n_experts, 1]
        weighted_outputs = outputs * gates  # [B, n_experts, C]
        
        return weighted_outputs.sum(dim=1)  # [B, C]



# ======================
# Train test
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"

trainset = ImageDataset('/home/lucas/MoE-PKLot/CSV/camera9/camera9_train.csv')
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

validset = ImageDataset('/home/lucas/MoE-PKLot/CSV/camera9/camera9_valid.csv')
validloader = DataLoader(validset, batch_size=64, shuffle=False)

testset = ImageDataset('/home/lucas/MoE-PKLot/CSV/camera9/camera9_test.csv')
testloader = DataLoader(testset, batch_size=64, shuffle=False)

model = MoECNN(n_experts=10, top_k=3).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Treinar por múltiplas épocas
num_epochs = 3

for epoch in range(num_epochs):
    # ===== TREINO =====
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.long().to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_acc = 100 * train_correct / train_total
    train_avg_loss = train_loss / len(trainloader)
    
    # ===== VALIDAÇÃO =====
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in validloader:
            images = images.to(device)
            labels = labels.long().to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_acc = 100 * val_correct / val_total
    val_avg_loss = val_loss / len(validloader)
    
    print(f"Época {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_avg_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

# ===== TESTE =====
print("\n" + "="*60)
print("AVALIAÇÃO NO CONJUNTO DE TESTE")
print("="*60)

model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.long().to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_acc = 100 * test_correct / test_total
test_avg_loss = test_loss / len(testloader)

print(f"Test Loss: {test_avg_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")
