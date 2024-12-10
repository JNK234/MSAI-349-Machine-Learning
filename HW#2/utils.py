import numpy as np
import math

# Distance Metrics
def euclidean(a, b):
    """Calculate Euclidean distance between two vectors."""
    return float(np.sqrt(np.sum((np.array(a) - np.array(b)) ** 2)))

def cosine_similarity(a, b):
    """Calculate Cosine Similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def pearson_correlation(a, b):
    """Calculate Pearson Correlation between two vectors."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return np.corrcoef(a, b)[0, 1]

def hamming_distance(a, b):
    """Calculate Hamming distance between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.sum(a != b)

# Data Loading Functions
def read_mnist_data(file_name):
    """Read MNIST data from CSV file."""
    data_set = []
    with open(file_name, 'rt') as f:
        for line in f:
            line = line.replace('\n', '')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label, attribs])
    return data_set

def preprocess_data_quantized(data, levels=4):
    """Preprocess data by quantizing into specified number of levels."""
    processed_data = []
    for label, attributes in data:
        float_attributes = [float(x) for x in attributes]
        quantized_attributes = [int(x / (256 / levels)) for x in float_attributes]
        processed_data.append([label, quantized_attributes])
    return processed_data