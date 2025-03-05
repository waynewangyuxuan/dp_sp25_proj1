import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Optional

class CIFAR10Dataset(Dataset):
    """CIFAR-10 Dataset"""
    
    def __init__(self, root_dir, transform=None, is_train=True, batch_files=None):
        """
        Args:
            root_dir (string): Directory with all the CIFAR-10 batches
            transform (callable, optional): Optional transform to be applied on a sample
            is_train (bool): If True, creates dataset from training set, otherwise from test set
            batch_files (List[str], optional): List of batch files to use (e.g., ['data_batch_1', 'data_batch_2'])
                                              If None and is_train=True, all training batches are used
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.batch_files = batch_files
        
        if self.is_train:
            # Load training data
            self.data = []
            self.targets = []
            
            # Determine which batch files to use
            if batch_files is None:
                # Use all training batches by default
                batch_files = [f'data_batch_{i}' for i in range(1, 6)]
            
            # Load specified batch files
            for batch_file in batch_files:
                file_path = os.path.join(self.root_dir, batch_file)
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
                test_data = pickle.load(f, encoding='latin1')  # Load raw data
                
                # Print test data keys for debugging
                print(f"Test data type: {type(test_data)}")
                if isinstance(test_data, dict):
                    print(f"Test data keys: {test_data.keys()}")
                
                # Handle different test data formats
                if isinstance(test_data, dict):
                    # Try different possible keys
                    if 'images' in test_data:
                        self.data = test_data['images']
                    elif 'data' in test_data:
                        self.data = test_data['data']
                    else:
                        # Just use the first key if none of the expected keys are found
                        first_key = list(test_data.keys())[0]
                        self.data = test_data[first_key]
                        print(f"Using key '{first_key}' for test data")
                elif isinstance(test_data, np.ndarray):
                    self.data = test_data
                else:
                    raise ValueError(f"Unexpected test data format: {type(test_data)}")
                
                # Ensure data is a numpy array
                self.data = np.array(self.data)
                
                # Reshape data if needed
                if len(self.data.shape) == 2:  # If data is flattened
                    self.data = self.data.reshape(-1, 3, 32, 32)
                    self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
                
                # Create dummy targets (indices)
                self.targets = np.arange(len(self.data))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        
        # Convert to PIL Image for transforms that require it
        if self.transform and isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
            
        if self.is_train:
            target = self.targets[idx]
            return image, target
        else:
            return image, idx  # Return index for test set 