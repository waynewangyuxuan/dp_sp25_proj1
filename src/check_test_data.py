import pickle
import numpy as np

# Load test data
with open('data/cifar10/cifar_test_nolabel.pkl', 'rb') as f:
    test_data = pickle.load(f, encoding='latin1')

# Print information about the data
print("Type of test_data:", type(test_data))
if isinstance(test_data, dict):
    print("Keys:", list(test_data.keys()))
    for key in test_data:
        if isinstance(test_data[key], np.ndarray):
            print(f"{key} shape:", test_data[key].shape)
        else:
            print(f"{key} type:", type(test_data[key]))
else:
    print("Data structure:", test_data) 