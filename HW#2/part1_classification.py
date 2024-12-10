import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report, adjusted_rand_score
from utils import *
import random

class MNISTClassifier:
    def __init__(self, k=5):
        self.k = k

    def knn_classify(self, train, query, metric="euclidean"):
        """K-Nearest Neighbors classification."""
        labels = []
        
        for q in query:
            distances = []
            for t in train:
                label = t[0]
                attributes = np.array(t[1], dtype=float)
                
                if metric == "euclidean":
                    dist = euclidean(attributes, q)
                elif metric == "cosine":
                    dist = 1 - cosine_similarity(attributes, q)
                elif metric == "pearson":
                    dist = 1 - pearson_correlation(attributes, q)
                elif metric == "hamming":
                    dist = hamming_distance(attributes, q)
                    
                distances.append((dist, label))
            
            # Sort distances and get k nearest neighbors
            distances.sort(key=lambda x: x[0])
            nearest_labels = [d[1] for d in distances[:self.k]]
            
            # Majority voting
            label_count = {}
            for label in nearest_labels:
                label_count[label] = label_count.get(label, 0) + 1
            
            predicted_label = max(label_count.items(), key=lambda x: x[1])[0]
            labels.append(predicted_label)
            
        return labels

    def kmeans_cluster(self, train, k=10, metric="euclidean", max_iters=100):
        """K-Means clustering."""
        # Initialize centroids randomly
        centroids = random.sample(train, k)
        centroids = [np.array(c[1], dtype=float) for c in centroids]
        
        for _ in range(max_iters):
            # Assign points to clusters
            clusters = [[] for _ in range(k)]
            for t in train:
                distances = []
                for centroid in centroids:
                    if metric == "euclidean":
                        dist = euclidean(t[1], centroid)
                    elif metric == "cosine":
                        dist = 1 - cosine_similarity(t[1], centroid)
                    elif metric == "pearson":
                        dist = 1 - pearson_correlation(t[1], centroid)
                    elif metric == "hamming":
                        dist = hamming_distance(t[1], centroid)
                    distances.append(dist)
                    
                closest_centroid = np.argmin(distances)
                clusters[closest_centroid].append(t)
            
            # Update centroids
            new_centroids = []
            for cluster in clusters:
                if cluster:
                    new_centroid = np.mean([np.array(c[1], dtype=float) for c in cluster], axis=0)
                else:
                    new_centroid = np.array(random.choice(train)[1], dtype=float)
                new_centroids.append(new_centroid)
            
            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
                
            centroids = new_centroids
            
        return centroids, clusters

    def evaluate_classifier(self, predictions, true_labels):
        """Evaluate classifier performance."""
        # Confusion Matrix
        cm = confusion_matrix(true_labels, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Confusion Matrix")
        plt.show()
        
        # Calculate and print metrics
        accuracy = accuracy_score(true_labels, predictions)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions))
        
        return accuracy

def main():
    # Load and preprocess data
    train_data = preprocess_data_quantized(read_mnist_data('mnist_train.csv'))
    test_data = preprocess_data_quantized(read_mnist_data('mnist_valid.csv'))
    
    # Prepare data
    train_labels = [x[0] for x in train_data]
    train_attributes = [x[1] for x in train_data]
    test_labels = [x[0] for x in test_data]
    test_attributes = [x[1] for x in test_data]
    
    # Initialize classifier
    classifier = MNISTClassifier(k=5)
    
    # Test different distance metrics for KNN
    metrics = ["euclidean", "cosine", "pearson", "hamming"]
    results = {}
    
    print("KNN Classification Results:")
    print("-" * 50)
    
    for metric in metrics:
        print(f"\nTesting {metric.capitalize()} Distance:")
        predictions = classifier.knn_classify(
            list(zip(train_labels, train_attributes)),
            test_attributes,
            metric
        )
        accuracy = classifier.evaluate_classifier(predictions, test_labels)
        results[metric] = accuracy
    
    # K-means clustering
    print("\nK-means Clustering Results:")
    print("-" * 50)
    
    k = 10  # Number of clusters (one for each digit)
    for metric in metrics:
        print(f"\nTesting {metric.capitalize()} Distance:")
        centroids, clusters = classifier.kmeans_cluster(train_data, k=k, metric=metric)
        
        # Assign labels to clusters based on majority voting
        cluster_labels = []
        for cluster in clusters:
            if cluster:
                labels = [point[0] for point in cluster]
                most_common = max(set(labels), key=labels.count)
                cluster_labels.append(most_common)
        
        # Evaluate clustering using adjusted Rand index
        predicted_labels = []
        for point in test_data:
            distances = []
            for centroid in centroids:
                if metric == "euclidean":
                    dist = euclidean(point[1], centroid)
                elif metric == "cosine":
                    dist = 1 - cosine_similarity(point[1], centroid)
                elif metric == "pearson":
                    dist = 1 - pearson_correlation(point[1], centroid)
                elif metric == "hamming":
                    dist = hamming_distance(point[1], centroid)
                distances.append(dist)
            cluster_idx = np.argmin(distances)
            if cluster_idx < len(cluster_labels):
                predicted_labels.append(cluster_labels[cluster_idx])
            else:
                predicted_labels.append(random.choice(cluster_labels))
        
        ari = adjusted_rand_score(test_labels, predicted_labels)
        print(f"Adjusted Rand Index: {ari:.4f}")

if __name__ == "__main__":
    main()