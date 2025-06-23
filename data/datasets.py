import os
import re
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

# === 通用配置 ===
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# === 统一标签映射 ===
LABEL_MAP = {
    'angry': 0, 'anger': 0, 'contempt': 1, 'disgust': 2, 'fear': 3,
    'happy': 4, 'sad': 5, 'sadness': 5, 'surprise': 6, 'neutral': 7
}

# === 数据集基类 ===
class BaseFERDataset(Dataset):
    def __init__(self, samples, transform=None, label_map=None):
        self.samples = samples
        self.transform = transform
        self.label_map = label_map or LABEL_MAP
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        if len(item) == 3:  
            img_path, label, subject_id = item
        else:
            img_path, label = item[:2]
            subject_id = None
        
        try:
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            print(f"无法加载图像: {img_path}, 错误: {e}")
            # 返回一个空图像作为占位符
            image = Image.new('RGB', (128, 128), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        sample = {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }
        if subject_id is not None:
            sample['subject_id'] = subject_id
        return sample

# === CK+ 特定函数和类 ===
def extract_subject_id(img_path):
    """提取 CK+ 图片中的被试 ID"""
    filename = os.path.basename(img_path)
    # 支持 S001_001_00000012.png 或 S001-001-00000012.png
    match = re.match(r"([S]\d+)[_\-]\d+[_\-]\d+\.png", filename)
    if match:
        return match.group(1)
    return None

class CKPlusDataset(BaseFERDataset):
    """CK+ 数据集专用类"""
    def __init__(self, samples=None, transform=None):
        super().__init__(samples, transform, LABEL_MAP)
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if len(self.samples[idx]) >= 3:
            sample['subject_id'] = self.samples[idx][2]
        return sample

# === 数据集加载函数 ===
def load_fer2013(config):
    """加载 FER2013 数据集（带验证集分割）"""
    root_dir = config['root_dir']
    subset = config.get('subset', 'train')
    transform = config.get('transform', get_default_transforms('fer2013'))
    val_split = config.get('val_split', 0.176)
    samples = []
    data_dir = os.path.join(root_dir, subset)
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            label = LABEL_MAP[class_name.lower()]
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    samples.append((
                        os.path.join(class_dir, img_file),
                        label
                    ))
    
    #划分验证集
    if subset == 'train' and val_split > 0:
        labels = [s[1] for s in samples]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, 
                                    random_state=RANDOM_SEED)
        train_idx, val_idx = next(sss.split(np.zeros(len(samples)), labels))
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        return (
            BaseFERDataset(train_samples, transform),
            BaseFERDataset(val_samples, transform),
            BaseFERDataset(samples, transform) 
        )
    else:
        return BaseFERDataset(samples, transform)

def load_ckplus(config):
    """加载 CK+ 数据集"""
    root_dir = config['root_dir']
    transform = config.get('transform', get_default_transforms('ckplus'))
    test_size = config.get('test_size', 0.3)
    all_samples = []
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        #CK+ 的 contempt 表情类映射
        label = LABEL_MAP[class_name.lower()] if class_name.lower() != 'contempt' else LABEL_MAP['contempt']
        
        for img_file in os.listdir(class_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(class_dir, img_file)
            subject_id = extract_subject_id(img_path) or f'unknown_{len(all_samples)}'
            all_samples.append((img_path, label, subject_id))
    
    #按被试划分数据集
    subject_ids = list(set(s[2] for s in all_samples))
    subject_labels = [np.random.choice([s[1] for s in all_samples if s[2] == sid]) 
                     for sid in subject_ids]
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, 
                                random_state=RANDOM_SEED)
    train_sub_ids, test_sub_ids = next(sss.split(np.zeros(len(subject_ids)), subject_labels))
    #再划分验证集
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, 
                                    random_state=RANDOM_SEED)
    val_sub_ids, test_sub_ids = next(sss_val.split(test_sub_ids, 
                                                  [subject_labels[i] for i in test_sub_ids]))
    #收集样本
    train_samples = [s for s in all_samples if s[2] in [subject_ids[i] for i in train_sub_ids]]
    val_samples = [s for s in all_samples if s[2] in [subject_ids[i] for i in val_sub_ids]]
    test_samples = [s for s in all_samples if s[2] in [subject_ids[i] for i in test_sub_ids]]
    return (
        CKPlusDataset(train_samples, transform),
        CKPlusDataset(val_samples, transform),
        CKPlusDataset(test_samples, transform)
    )

def load_rafdb(config):
    """加载 RAF-DB 数据集（带验证集分割）"""
    csv_path = config['csv_path']
    root_dir = config['root_dir']
    transform = config.get('transform', get_default_transforms('rafdb'))
    val_split = config.get('val_split', 0.176)
    df = pd.read_csv(csv_path, sep=' ', header=None, names=['path', 'label'])
    df['path'] = df['path'].str.replace('.jpg', '_aligned.jpg')
    df['full_path'] = df['path'].apply(lambda x: os.path.join(root_dir, x))
    # 完整数据集
    full_set = [(row['full_path'], row['label']) for _, row in df.iterrows()]
    # 划分训练验证集
    if val_split > 0:
        labels = df['label'].values
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, 
                                    random_state=RANDOM_SEED)
        train_idx, val_idx = next(sss.split(np.zeros(len(df)), labels))
        train_samples = [full_set[i] for i in train_idx]
        val_samples = [full_set[i] for i in val_idx]
        return (
            BaseFERDataset(train_samples, transform),
            BaseFERDataset(val_samples, transform),
            BaseFERDataset(full_set, transform)
        )
    else:
        return BaseFERDataset(full_set, transform)

# === 辅助函数 ===
def get_default_transforms(dataset_name):
    """获取各数据集的默认数据增强变换"""
    base_size = 128
    train_transform = transforms.Compose([
        transforms.Resize((base_size, base_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((base_size, base_size)),
        transforms.ToTensor(),
    ])
    if dataset_name == 'fer2013':
        mean, std = [0.485], [0.229]
    else:  #CK+ 和 RAF-DB
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    return transforms.Compose([
        *train_transform.transforms, 
        transforms.Normalize(mean, std)
    ]), transforms.Compose([
        *val_test_transform.transforms, 
        transforms.Normalize(mean, std)
    ])
