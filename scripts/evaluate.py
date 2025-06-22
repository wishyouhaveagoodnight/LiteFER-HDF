import torch
import numpy as np
import yaml
import os
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.LiteFER_HDF import LiteFER_HDF
from models.datasets import load_dataset
from utils.metrics import evaluate_model, calculate_metrics
from utils.visualization import plot_confusion_matrix
def evaluate(config_path, model_path):
    """评估模型"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and 
        config['training']['device'] == 'cuda' else "cpu"
    )
    print(f"使用设备: {device}")
    _, _, test_dataset = load_dataset(config)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    #模型
    model_class = {
        'LiteFER_HDF': LiteFER_HDF
    }[config['model']['type']]
    model = model_class(
        num_classes=config['model']['num_classes'],
        alpha=config['model']['alpha'],
        d_model=config['model']['d_model']
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"加载模型: {model_path}")
    #评估模型
    true_labels, pred_labels = evaluate_model(model, device, test_loader)
    #计算评估指标
    classes = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    metrics = calculate_metrics(true_labels, pred_labels, classes)
    print("\n评估结果:")
    print(f"测试准确率: {metrics['accuracy'] * 100:.2f}%")
    print(f"宏平均精确率: {metrics['precision']:.4f}")
    print(f"宏平均召回率: {metrics['recall']:.4f}")
    print(f"宏平均F1分数: {metrics['f1']:.4f}")
    # 打印每类指标
    print("\n每类性能:")
    for class_name, class_metric in metrics['class_metrics'].items():
        print(f"{class_name}: 精确率={class_metric['precision']:.4f}, "
              f"召回率={class_metric['recall']:.4f}, F1={class_metric['f1']:.4f}")
    #可视化
    dataset_name = config['dataset']['name']
    save_path = f"results/{dataset_name}_confusion_matrix.png"
    plot_confusion_matrix(
        metrics['confusion_matrix'], 
        classes,
        title=f"{dataset_name}测试集混淆矩阵",
        save_path=save_path
    )
    print(f"混淆矩阵保存至: {save_path}")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../configs/default.yaml', 
                        help='配置文件路径')
    parser.add_argument('--model', default='checkpoints/best_model.pth', 
                        help='模型路径')
    args = parser.parse_args()
    evaluate(args.config, args.model)
