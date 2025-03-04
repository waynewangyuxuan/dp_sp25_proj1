import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class CIFAR10Dataset(Dataset):
    """CIFAR-10 Dataset"""
    
    def __init__(self, root_dir, transform=None, is_train=True):
        """
        Args:
            root_dir (string): Directory with all the CIFAR-10 batches
            transform (callable, optional): Optional transform to be applied on a sample
            is_train (bool): If True, creates dataset from training set, otherwise from test set
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        
        if self.is_train:
            # Load training data
            self.data = []
            self.targets = []
            for i in range(1, 6):
                file_path = os.path.join(self.root_dir, f'data_batch_{i}')
                with open(file_path, 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                    self.data.append(entry['data'])
                    self.targets.extend(entry['labels'])
            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
            
            # Load label names
            meta_file = os.path.join(self.root_dir, 'batches.meta')
            with open(meta_file, 'rb') as f:
                meta = pickle.load(f, encoding='latin1')
                self.label_names = meta['label_names']
        else:
            # Load test data - adjust path for test file
            test_file = os.path.join(os.path.dirname(os.path.dirname(self.root_dir)), 'cifar_test_nolabel.pkl')
            with open(test_file, 'rb') as f:
                self.data = pickle.load(f, encoding='latin1')  # Load raw data
                if isinstance(self.data, dict):
                    self.data = self.data['images']  # Try 'images' key if it's a dict
                self.data = np.array(self.data).reshape(-1, 3, 32, 32)
                self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
                self.targets = None  # No labels for test set
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        
        if self.transform:
            image = self.transform(image)
            
        if self.is_train:
            target = self.targets[idx]
            return image, target
        else:
            return image, idx  # Return index for test set 