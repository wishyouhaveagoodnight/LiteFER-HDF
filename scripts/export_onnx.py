import torch
import yaml

from models.LiteFER-HDF import LiteFER-HDF

#加载配置
with open('../configs/default.yaml') as f:
    config = yaml.safe_load(f)

#加载模型
model = LiteFER-HDF(
    num_classes=config['model']['num_classes'],
    alpha=config['model']['alpha'],
    d_model=config['model']['d_model']
)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

#导出ONNX
dummy_input = torch.randn(1, 3, 128, 128)
torch.onnx.export(
    model, 
    dummy_input,
    "litefer.onnx",
    opset_version=13,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}}
)
