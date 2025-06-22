import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_training_history(history, test_acc=None):
    """绘制训练历史"""
    plt.figure(figsize=(15, 10))
    #损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history.get('train_loss', []), label='训练损失')
    plt.plot(history.get('val_loss', []), label='验证损失')
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    plt.title('训练和验证损失')
    plt.legend()
    #准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(history.get('train_acc', []), label='训练准确率')
    plt.plot(history.get('val_acc', []), label='验证准确率')
    if test_acc is not None:
        plt.axhline(y=test_acc, color='r', linestyle='--', 
                    label=f'测试准确率 ({test_acc:.2f}%)')
    plt.xlabel('迭代次数')
    plt.ylabel('准确率 (%)')
    plt.title('训练和验证准确率')
    plt.legend()
    #学习率曲线
    plt.subplot(2, 2, 3)
    plt.plot(history.get('lr', []))
    plt.xlabel('迭代次数')
    plt.ylabel('学习率')
    plt.title('学习率变化')
    plt.yscale('log')
    #F1值曲线
    plt.subplot(2, 2, 4)
    if 'train_f1' in history:
        plt.plot(history['train_f1'], label='训练F1')
    if 'val_f1' in history:
        plt.plot(history['val_f1'], label='验证F1')
    plt.xlabel('迭代次数')
    plt.ylabel('F1值')
    plt.title('训练和验证F1值')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, 
               yticklabels=class_names)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
def plot_feature_maps(features, n_cols=6, save_path='feature_maps.png'):
    """可视化模型的特征图"""
    n_features = features.shape[1]
    n_rows = (n_features + n_cols - 1) // n_cols
    plt.figure(figsize=(2 * n_cols, 2 * n_rows))
    for i in range(n_features):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(features[0, i].detach().cpu().numpy(), cmap='viridis')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
