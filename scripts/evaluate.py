import torch
from torch.utils.data import DataLoader
import yaml

from models.LiteFER-HDF import LiteFER-HDF
from models.datasets import load_fer2013, load_ckplus, load_rafdb
from utils.metrics import evaluate_model, plot_confusion_matrix

# 加载配置
with open('../configs/default.yaml') as f:
    config = yaml.safe_load(f)

# 加载模型
model = LiteFER-HDF(
    num_classes=config['model']['num_classes'],
    alpha=config['model']['alpha'],
    d_model=config['model']['d_model']
)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 加载测试集
test_dataset = 'fer2013'  # 根据数据集类型加载'fer2013'、'ckplus'、'rafdb'
test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'])

# 评估
true_labels, pred_labels = evaluate_model(model, device, test_loader)
classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
plot_confusion_matrix(true_labels, pred_labels, classes)
