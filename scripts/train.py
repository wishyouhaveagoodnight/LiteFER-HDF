import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os

from models.LiteFER-HDF import LiteFER-HDF
from models.datasets import load_fer2013, load_ckplus, load_rafdb
from utils.metrics import evaluate_model
from utils.visualization import plot_training_history

#加载配置
with open('../configs/default.yaml') as f:
    config = yaml.safe_load(f)

device = torch.device(
    "cuda" if torch.cuda.is_available() and 
    config['training']['device'] == 'cuda' else "cpu"
)

#数据集加载
dataset_name = config['dataset']['name']

if dataset_name == 'fer2013':
    train_dataset = load_fer2013(
        config['dataset']['root_dir'], 
        subset='train',
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),  
        ])
    )
elif dataset_name == 'ckplus':
    train_dataset, test_dataset = load_ckplus(
        config['dataset']['root_dir'],
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]  
            )
        ])
    )
elif dataset_name == 'rafdb':
    train_dataset = load_rafdb(
        config['dataset']['csv_path'],
        config['dataset']['root_dir'],
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )
    
#划分测试集
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

#模型初始化
model = LiteFER-HDF(
    num_classes=config['model']['num_classes'],
    alpha=config['model']['alpha'],
    d_model=config['model']['d_model']
).to(device)

#训练循环
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(), 
    lr=config['training']['lr'],
    weight_decay=config['training']['weight_decay']
)

train_losses, val_accuracies = [], []

for epoch in range(config['training']['epochs']):
    model.train()
    epoch_loss = 0.0
    
    for batch in train_loader:
        inputs = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    #评估部分
    train_losses.append(epoch_loss / len(train_loader))
    labels, preds = evaluate_model(model, device, test_loader)
    accuracy = accuracy_score(labels, preds)
    val_accuracies.append(accuracy)
    
    print(f'Epoch {epoch+1}/{config["training"]["epochs"]}')
    print(f'Train Loss: {train_losses[-1]:.4f} | Val Acc: {accuracy:.4f}')

#保存模型
torch.save(model.state_dict(), 'best_model.pth')
plot_training_history(train_losses, val_accuracies)
