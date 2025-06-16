import matplotlib.pyplot as plt

def plot_training_history(train_losses, val_accuracies):
    """绘制训练历史"""
    fig, ax1 = plt.subplots(figsize=(10, 5))
    #训练损失
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(train_losses, color='tab:red', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    #验证准确率
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:blue')
    ax2.plot(val_accuracies, color='tab:blue', label='Val Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    fig.tight_layout()
    fig.legend(loc='upper right')
    plt.title('Training History')
    plt.show()
