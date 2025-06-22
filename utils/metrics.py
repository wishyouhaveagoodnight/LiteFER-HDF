import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, device, data_loader):
    """评估模型"""
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            labels = batch['label'].cpu().numpy()
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            true_labels.extend(labels)
            pred_labels.extend(predicted.cpu().numpy())
    return np.array(true_labels), np.array(pred_labels)

def calculate_metrics(true_labels, pred_labels, classes=None):
    """计算多种评估指标"""
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='macro')
    recall = recall_score(true_labels, pred_labels, average='macro')
    f1 = f1_score(true_labels, pred_labels, average='macro')
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    if classes is not None:
        cm = confusion_matrix(true_labels, pred_labels)
        metrics['confusion_matrix'] = cm
        #计算每类精确率
        class_precision = precision_score(true_labels, pred_labels, average=None)
        #计算每类召回率
        class_recall = recall_score(true_labels, pred_labels, average=None)
        #计算每类F1分数
        class_f1 = f1_score(true_labels, pred_labels, average=None)
        class_metrics = {}
        for i, class_name in enumerate(classes):
            class_metrics[class_name] = {
                'precision': class_precision[i],
                'recall': class_recall[i],
                'f1': class_f1[i]
            }
        metrics['class_metrics'] = class_metrics
    return metrics
