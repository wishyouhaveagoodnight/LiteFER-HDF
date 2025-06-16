import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
#标签映射
LABEL_MAP = {
    'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
    'sad': 4, 'surprise': 5, 'neutral': 6
}
class BaseFERDataset(Dataset):
    """表情识别数据集基类"""
    def __init__(self, samples, transform=None, label_map=None):
        self.samples = samples
        self.transform = transform
        self.label_map = label_map or LABEL_MAP
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_fer2013(root_dir, subset='train', transform=None):
    """加载FER2013数据集"""
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
    
    return BaseFERDataset(samples, transform)

def load_ckplus(root_dir, transform=None, test_size=0.2):
    """CK+数据集"""
    samples = []
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            label = LABEL_MAP[class_name.lower()]
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    samples.append((
                        os.path.join(class_dir, img_file),
                        label
                    ))
    
    train_samples, test_samples = train_test_split(
        samples, test_size=test_size, random_state=42
    )
    
    return (
        BaseFERDataset(train_samples, transform),
        BaseFERDataset(test_samples, transform)
    )

def load_rafdb(csv_path, root_dir, transform=None):
    """RAF-DB数据集"""
    df = pd.read_csv(csv_path, sep=' ', header=None, names=['path', 'label'])
    df['path'] = df['path'].apply(
        lambda x: x.replace('.jpg', '_aligned.jpg')
    )
    
    return BaseFERDataset(
        [(os.path.join(root_dir, row['path']), row['label']) 
         for _, row in df.iterrows()],
        transform
    )
