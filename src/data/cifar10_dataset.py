import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class CIFAR10Dataset(Dataset):
    """CIFAR-10 Dataset"""
    
    def __init__(self, data_dir, batch_files, transform=None, is_test=False):
        """
        Args:
            data_dir: Directory containing the CIFAR-10 batch files
            batch_files: List of batch files to load (e.g., ['data_batch_1', 'data_batch_2'])
            transform: Optional transform to be applied on a sample
            is_test: If True, loads the unlabeled test dataset
        """
        self.data_dir = data_dir
        self.transform = transform
        self.is_test = is_test
        self.data = []
        self.targets = []
        
        if not is_test:
            # Load training/validation data from batch files
            for batch_file in batch_files:
                file_path = os.path.join(data_dir, batch_file)
                with open(file_path, 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                    self.data.append(entry['data'])
                    self.targets.extend(entry['labels'])
            
            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
            self.data = self.data.transpose((0, 2, 3, 1))  # Convert to HWC format
            
            # Load label names
            meta_file = os.path.join(data_dir, 'batches.meta')
            with open(meta_file, 'rb') as f:
                meta = pickle.load(f, encoding='latin1')
                self.label_names = meta['label_names']
        else:
            # Load unlabeled test data
            # Go up two directories from data_dir to reach the cifar10 directory
            base_dir = os.path.dirname(os.path.dirname(data_dir))
            test_file = os.path.join(base_dir, 'cifar_test_nolabel.pkl')
            with open(test_file, 'rb') as f:
                test_data = pickle.load(f, encoding='latin1')
                self.data = test_data[b'data']  # Already in HWC format
                self.image_ids = test_data[b'ids']  # Store image IDs for submission
                # For test set, we'll use dummy targets
                self.targets = [-1] * len(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx] if not self.is_test else -1
        
        # Convert to PIL Image
        img = Image.fromarray(img)
        
        # Convert to tensor and normalize
        img = torch.tensor(np.array(img)).float().permute(2, 0, 1) / 255.0
        
        if self.transform:
            img = self.transform(img)
        
        if self.is_test:
            # For test set, return image and its ID (useful for submission)
            return img, self.image_ids[idx]
        return img, target 