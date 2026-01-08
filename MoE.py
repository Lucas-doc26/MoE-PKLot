import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
import os
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# Dataset
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



# MoE
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
        
        # Aplica gates (ponderacao dos experts)
        gates = gates.unsqueeze(-1)  # [B, n_experts, 1]
        weighted_outputs = outputs * gates  # [B, n_experts, C]
        
        return weighted_outputs.sum(dim=1)  # [B, C]



def main():
    # Argument Parser
    parser = argparse.ArgumentParser(description='MoE (Mixture of Experts) para deteccao de vagas de estacionamento')
    
    parser.add_argument('--train_data', type=str, default='PUC', help='Caminho para dataset de treino (padrão: PUC)')
    parser.add_argument('--valid_data', type=str, default='PUC', help='Caminho para dataset de validacao (padrão: PUC)')
    parser.add_argument('--test_data', type=str, default='UFPR05', help='Caminho para dataset de teste (padrão: UFPR05)')
    parser.add_argument('--test_datasets', type=str, default='PUC,UFPR04,UFPR05,camera1,camera2,camera3,camera4,camera5,camera6,camera7,camera8,camera9,PKLot,CNR', help='Lista separada por vírgula de datasets para avaliar (ex: PUC,UFPR05,camera1,PKLot,CNR)')
    parser.add_argument('--batch_size', type=int, default=64, help='Tamanho do batch (padrão: 64)')
    parser.add_argument('--num_workers', type=int, default=1, help='Número de workers para DataLoader (padrão: 1)')
    parser.add_argument('--num_epochs', type=int, default=5, help='Número de épocas (padrão: 10)')
    parser.add_argument('--top_k', type=int, default=2, help='Top-k experts a selecionar (padrão: 2)')
    parser.add_argument('--n_experts', type=int, default=3, help='Número total de experts (padrão: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (padrão: 1e-3)')
    
    args = parser.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando device: {device}")
    
    # Criar diretório de modelos se não existir
    os.makedirs('Models', exist_ok=True)
    
    # Carregamento dos dados
    print("\n[1/5] Carregando datasets...")
    trainset = ImageDataset(f'/home/lucas.ocunha/MoE-PKLot/CSV/{args.train_data}/batches/batch-{args.batch_size}.csv')
    validset = ImageDataset(f'/home/lucas.ocunha/MoE-PKLot/CSV/{args.valid_data}/batches/batch-{args.batch_size}.csv')
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=args.num_workers)
    validloader = DataLoader(validset, batch_size=32, shuffle=False, num_workers=args.num_workers)
    # test datasets will be evaluated after training using --test_datasets
    
    print(f"   - Treino: {len(trainset)} imagens")
    print(f"   - Validacao: {len(validset)} imagens")
    print(f"   - Testes: múltiplos datasets (use --test_datasets)")
    
    # Modelo
    print("\n[2/5] Criando modelo MoE...")
    model = MoECNN(n_experts=args.n_experts, top_k=args.top_k).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"   - Experts: {args.n_experts}")
    print(f"   - Top-k: {args.top_k}")
    print(f"   - Learning rate: {args.lr}")
    
    # Treino
    print(f"\n[3/5] Iniciando treino ({args.num_epochs} épocas)...")
    
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(args.num_epochs):
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
        
        # ===== VALIDAcaO =====
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
        
        # Armazenar histórico
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_avg_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_avg_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Época {epoch+1}/{args.num_epochs} | "
              f"Train Loss: {train_avg_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_avg_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # Teste em múltiplos datasets
    print("\n[4/5] Avaliando modelo nos conjuntos de teste listados...")
    model.eval()

    def _find_test_csv(dataset_name):
        # procura por arquivos de teste dentro da pasta CSV/<dataset_name>
        base = '/home/lucas.ocunha/MoE-PKLot/CSV'
        parts = dataset_name.split('/') if '/' in dataset_name else [dataset_name]
        data_dir = os.path.join(base, *parts)
        if os.path.isdir(data_dir):
            # prioriza arquivos que contenham 'test' no nome
            for fname in os.listdir(data_dir):
                if 'test' in fname.lower() and fname.lower().endswith('.csv'):
                    return os.path.join(data_dir, fname)
            # fallback: primeiro CSV encontrado
            for fname in os.listdir(data_dir):
                if fname.lower().endswith('.csv'):
                    return os.path.join(data_dir, fname)
        # se recebeu um caminho absoluto
        if os.path.isfile(dataset_name):
            return dataset_name
        return None

    test_list = [t.strip() for t in args.test_datasets.split(',') if t.strip()]
    for test_ds in test_list:
        test_csv = _find_test_csv(test_ds)
        if test_csv is None:
            print(f"⚠️  Arquivo de teste não encontrado para dataset: {test_ds} - pulando")
            continue

        print(f"\n--- Avaliando: {test_ds} ({test_csv}) ---")
        testset = ImageDataset(test_csv)
        testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=args.num_workers)

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

        if test_total == 0:
            print(f"⚠️  Nenhuma amostra no dataset de teste: {test_ds}")
            continue

        test_acc = 100 * test_correct / test_total
        test_avg_loss = test_loss / len(testloader)

        print("=" * 60)
        print(f"RESULTADOS NO CONJUNTO DE TESTE: {test_ds}")
        print("=" * 60)
        print(f"Test Loss: {test_avg_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")

        # Salvar métricas desta avaliacao
        metrics_row = pd.DataFrame({
            'timestamp': [datetime.now().isoformat()],
            'n_experts': [args.n_experts],
            'top_k': [args.top_k],
            'batch_size': [args.batch_size],
            'num_workers': [args.num_workers],
            'num_epochs': [args.num_epochs],
            'learning_rate': [args.lr],
            'device': [device],
            'train_dataset': [args.train_data],
            'valid_dataset': [args.valid_data],
            'test_dataset': [test_ds],
            'final_train_loss': [train_avg_loss],
            'final_train_acc': [train_acc],
            'final_val_loss': [val_avg_loss],
            'final_val_acc': [val_acc],
            'test_loss': [test_avg_loss],
            'test_acc': [test_acc],
        })

        metrics_file = 'metrics.csv'
        if os.path.exists(metrics_file):
            existing_df = pd.read_csv(metrics_file)
            metrics_row = pd.concat([existing_df, metrics_row], ignore_index=True)

        metrics_row.to_csv(metrics_file, index=False)
        print(f"   ✓ Métricas desta avaliacao salvas em: {metrics_file}")
    
    print("\n[5/5] Salvando histórico e pesos do modelo...\n")
    
    # Salvar histórico completo de treino
    history_df = pd.DataFrame(history)
    history_file = f'history_E{args.n_experts}_K{args.top_k}.csv'
    history_df.to_csv(history_file, index=False)
    print(f"   ✓ Histórico de treino salvo em: {history_file}")
    
    # Salvar pesos do modelo
    model_name = f'Moe-{args.train_data}-B{args.batch_size}-E{args.n_experts}-K{args.top_k}-W{args.num_workers}.pth'
    model_path = os.path.join('Models', model_name)
    torch.save(model.state_dict(), model_path)
    print(f"   ✓ Pesos do modelo salvos em: {model_path}")
    
    print("\n" + "=" * 60)
    print("TREINO CONCLUÍDO COM SUCESSO!")
    print("=" * 60)


if __name__ == '__main__':
    main()
