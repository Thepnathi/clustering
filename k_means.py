from cluster import Cluster_Algorithm
from dataset_handler import Dataset_Handler
from plot import Plot_B_CUBED
from constant import Constant
from sys import maxsize
from math import sqrt
from random import randrange

class K_Means_Algorithm(Cluster_Algorithm):
    def __init__(self, label_dataset, feature_dataset, category):
        super().__init__(label_dataset, feature_dataset, category)

    def k_means(self, k: int):
        # Step 1: Initialisation phase
        representative = self.compute_random_cluster_representative(k) # Contains the k representative of clusters or points as the centroid of clusters
        convergence_steps = 0
        feature_belongs_to_cluster = []
        while True:
            convergence_steps += 1
            # Step 2: Assignment phase.  Assign each feature points to a cluster or representative
            feature_belongs_to_cluster_temp = []     # Stores the index of representative that match the index feature dataset
            for _, row in enumerate(self.feature_dataset):
                min_dist, closest_rep = maxsize, None
                for j in range(len(representative)):
                    distance = self.euclidean_distance(representative[j], row)
                    if distance < min_dist:
                        min_dist = distance
                        closest_rep = j
                feature_belongs_to_cluster_temp.append(closest_rep)
            # Step 3: Optimisation phase. Calculate new representative - mean of all item within the cluster
            representative = self.compute_new_cluster_representatives(representative, feature_belongs_to_cluster_temp)
            if feature_belongs_to_cluster_temp == feature_belongs_to_cluster:
                    break
            feature_belongs_to_cluster = feature_belongs_to_cluster_temp
        print(f"k-means algorithm with k={k} cluster took {convergence_steps} steps until converged.")
        return feature_belongs_to_cluster, representative

    # For each representative, calculate the mean value of each feature
    def compute_new_cluster_representatives(self, cluster_representative, feature_belongs_to_cluster):
        for i in range(len(cluster_representative)): # Loop through each cluster point or representative
            total_features_in_cluster = 0 # Stores the number of feature in current cluster
            new_representative = [0] * len(cluster_representative[i])
            for j in range(len(feature_belongs_to_cluster)): # Loop through each feature_cluster
                if feature_belongs_to_cluster[j] == i:  # A feature belongs to the current cluster point or rep
                    total_features_in_cluster += 1
                    for f in range(len(self.feature_dataset[j])): # We loop through the current feature and increment the new rep
                        new_representative[f] += self.feature_dataset[j][f]
            if total_features_in_cluster > 0: # Some cluster will have no objects, we do not change the cluster rep
                cluster_representative[i] = [dimension/total_features_in_cluster for dimension in new_representative] # Update the current representative with new mean features 
        return cluster_representative

    def euclidean_distance(self, vec1, vec2):
        total = 0
        for i in range(len(vec1)):
            total += pow(vec1[i]- vec2[i], 2)
        return sqrt(total)

if __name__ == "__main__":
    data_handler = Dataset_Handler()
    plot_tool = Plot_B_CUBED()
    data_handler.load_multiple_dataset("animals", "countries", "fruits", "veggies") # Load the four category data files and merge into one

    category = data_handler.get_label_categories()
    label_dataset = data_handler.get_label_dataset()
    feature_dataset = data_handler.get_feature_dataset()
    feature_normalised_dataset = data_handler.get_feature_normalised_dataset()

    print(f'\n{Constant.line}\nNormal dataset version\n{Constant.line}\n')

    cluster = K_Means_Algorithm(label_dataset, feature_dataset, category)
    precision_per_k, recall_per_k, f_score_per_k, k_values = [], [], [], []
    for i in range(1, 10):
        feature_belongs_to_cluster, cluster_representative = cluster.k_means(i)
        result = cluster.compute_B_CUBED(feature_belongs_to_cluster, cluster_representative)
        print(result)
        k_values.append(i)
        precision_per_k.append(result.precision)
        recall_per_k.append(result.recall)
        f_score_per_k.append(result.f_score)
    plot_tool.plot_b_cubed(precision_per_k, recall_per_k, f_score_per_k, k_values, "K-Means Cluster B-CUBED Measure")

    print(f'\n{Constant.line}\nNormalised L2 dataset version\n{Constant.line}\n')

    cluster = K_Means_Algorithm(label_dataset, feature_normalised_dataset, category)
    precision_per_k, recall_per_k, f_score_per_k, k_values = [], [], [], []
    for i in range(1, 10):
        feature_belongs_to_cluster, cluster_representative = cluster.k_means(i)
        result = cluster.compute_B_CUBED(feature_belongs_to_cluster, cluster_representative)
        print(result)
        k_values.append(i)
        precision_per_k.append(result.precision)
        recall_per_k.append(result.recall)
        f_score_per_k.append(result.f_score)
    plot_tool.plot_b_cubed(precision_per_k, recall_per_k, f_score_per_k, k_values, "K-Means Cluster B-CUBED Measure L2 Norm")