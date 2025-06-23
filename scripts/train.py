import torch
import argparse
import yaml
import os
from torch import nn, optim
from torch.utils.data import DataLoader
from models import LiteFER_HDF, LiteFER-HDF, LiteFER-HDF_NoTransformer, LiteFER-HDF_NoDeform, LiteFER-HDF_NoAttention
from data.datasets import load_dataset  
from utils.metrics import accuracy_score
from utils.visualization import plot_training_history
# === 训练函数实现 ===
class EarlyStopping:
    """早停机制类"""
    def __init__(self, patience=5, min_delta=0.0, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_loss = float('inf')
        self.best_acc = 0.0
        self.best_model_state = None
        self.early_stop = False
    def __call__(self, model, val_loss, val_acc):
        if val_loss < self.best_loss - self.min_delta or val_acc > self.best_acc + self.min_delta:
            self.best_loss = min(val_loss, self.best_loss)
            self.best_acc = max(val_acc, self.best_acc)
            if self.restore_best:
                self.best_model_state = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best and self.best_model_state is not None:
                    model.load_state_dict(self.best_model_state)
        return self.early_stop
#配置预设
DATASET_CONFIG = {
    'CK+': {'max_epoch': 25, 'min_delta': 0.001, 'patience': 5},
    'RAF-DB': {'max_epoch': 40, 'min_delta': 0.003, 'patience': 6},
    'Fer2013': {'max_epoch': 60, 'min_delta': 0.002, 'patience': 12},
}
def train_model(model, device, train_loader, val_loader, criterion, 
                optimizer, dataset_name, save_path='checkpoints'):
    """
    改进的训练函数，包含完整训练监控和优化
    """
    cfg = DATASET_CONFIG[dataset_name]
    os.makedirs(save_path, exist_ok=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    early_stopper = EarlyStopping(
        patience=cfg['patience'], 
        min_delta=cfg['min_delta'],
        restore_best=True
    )
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    best_val_acc = 0.0
    for epoch in range(1, cfg['max_epoch'] + 1):
        model.train()
        train_loss = 0.0
        train_total = 0
        train_correct = 0
        #训练批次
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            #进度更新
            if batch_idx % max(1, len(train_loader)//10) == 0:
                print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        #验证阶段
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss, val_acc = validate(model, device, val_loader, criterion)
        #学习率调整
        scheduler.step(val_acc)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f'Epoch {epoch} 训练完成 | '
              f'训练损失: {avg_train_loss:.4f} | 训练准确率: {train_acc:.2f}% | '
              f'验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%')
        #最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_path, f'best_{dataset_name}.pth'))
            print(f'** 保存最佳模型! 准确率: {val_acc:.2f}% **')
        #早停检查
        if early_stopper(model, val_loss, val_acc):
            print(f"== 早停触发: 连续 {early_stopper.counter} 个epoch无明显改善 ==")
            print(f"最佳验证准确率: {early_stopper.best_acc:.2f}%")
            break
    
    print(f'训练完成! 最佳验证准确率: {best_val_acc:.2f}%')
    return best_val_acc, history
def validate(model, device, val_loader, criterion):
    """验证模型"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100. * correct / total
    return avg_val_loss, val_acc

def test(model, device, test_loader, criterion):
    """测试模型"""
    test_loss, test_acc = validate(model, device, test_loader, criterion)
    print(f'\n测试结果 - 损失: {test_loss:.4f}, 准确率: {test_acc:.2f}%')
    return test_loss, test_acc

# === 主程序 ===
def main(config_path):
    #加载配置
    with open(config_path) as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset, test_dataset = load_dataset(config['dataset'])
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    #模型初始化
    model_config = {
        'LiteFER-HDF': LiteFER_HDF,
        'LiteFER-HDF': LiteFER-HDF,
        'LiteFER-HDF_NoTransformer': LiteFER-HDF_NoTransformer,
        'LiteFER-HDF_NoDeform': LiteFER-HDF_NoDeform,
        'LiteFER-HDF_NoAttention': LiteFER-HDF_NoAttention
    }[config['model']['type']]
    model = model_config(
        num_classes=config['model']['num_classes'],
        alpha=config['model'].get('alpha', 0.75)
    ).to(device)
    #初始化优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    #训练模型
    best_val_acc, history = train_model(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        dataset_name=config['dataset']['name'],
        save_path=config['training']['save_path']
    )
    #最终测试
    model.load_state_dict(torch.load(os.path.join(
        config['training']['save_path'], 
        f'best_{config["dataset"]["name"]}.pth'
    )))
    test_loss, test_acc = test(model, device, test_loader, criterion)
    #可视化
    plot_training_history(history, test_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../configs/default.yaml', 
                        help='配置文件路径')
    args = parser.parse_args()
    main(args.config)
