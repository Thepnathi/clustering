from cluster import Cluster
from dataset_handler import Dataset_Handler
from constant import Constant
from sys import maxsize
from random import randrange

class K_Means(Cluster):
    def __init__(self, label_dataset, feature_dataset, category):
        super().__init__(label_dataset, feature_dataset, category)

    def k_means(self, feature_dataset, k: int):
        # Step 1: Initialisation phase
        representative = [] # Contains the k representative of clusters or points as the centroid of clusters
        for _ in range(k):
            random_representative = feature_dataset[randrange(len(feature_dataset))]  
            representative.append(random_representative) # What happens if we have the same representative?
        # Repeat until convergence - (no object has moved or by specified n iteration)
        convergence_steps = 0
        feature_belongs_to_cluster = []
        while True:
            convergence_steps += 1
            # Step 2: Assignment phase.  Assign each feature points to a cluster or representative
            feature_belongs_to_cluster_temp = []     # Stores the index of representative that match the index feature dataset
            for _, row in enumerate(feature_dataset):
                min_dist, closest_rep = maxsize, None
                for j in range(len(representative)):
                    distance = self.euclidean_distance(representative[j], row)
                    if distance < min_dist:
                        min_dist = distance
                        closest_rep = j
                feature_belongs_to_cluster_temp.append(closest_rep)
            # Step 3: Optimisation phase. Calculate new representative - mean of all item within the cluster
            representative = self.compute_new_representative(representative, feature_belongs_to_cluster_temp, feature_dataset)
            if feature_belongs_to_cluster_temp == feature_belongs_to_cluster:
                    break
            feature_belongs_to_cluster = feature_belongs_to_cluster_temp
        print(f"k-means algorithm with k={k} cluster took {convergence_steps} steps until converged.")
        return feature_belongs_to_cluster, representative

    # For each representative, calculate the mean value of each feature
    def compute_new_representative(self, representative, feature_belongs_to_cluster_temp, feature_dataset):
        for i in range(len(representative)): # Loop through each cluster point or representative
            total_features_in_cluster = 0 # Stores the number of feature in current cluster
            new_representative = [0] * len(representative[i])
            for j in range(len(feature_belongs_to_cluster_temp)): # Loop through each feature_cluster
                if feature_belongs_to_cluster_temp[j] == i:  # A feature belongs to the current cluster point or rep
                    total_features_in_cluster += 1
                    for f in range(len(feature_dataset[j])): # We loop through the current feature and increment the new rep
                        new_representative[f] += feature_dataset[j][f]
            representative[i] = [point/total_features_in_cluster for point in new_representative] # Update the current representative with new mean features 
        return representative

if __name__ == "__main__":
    data_handler = Dataset_Handler()
    data_handler.load_multiple_dataset("animals", "countries", "fruits", "veggies") # Load the four category data files and merge into one

    category = data_handler.get_label_categories()
    label_dataset, feature_dataset = data_handler.get_label_dataset(), data_handler.get_feature_dataset()

    cluster = K_Means(label_dataset, feature_dataset, category)

    for i in range(4, 5):
        feature_belongs_to_cluster, representative = cluster.k_means(feature_dataset, i)
        print(f'{Constant.line}\nComputing precision for k={i}\n{Constant.line}')
        precision = cluster.compute_precision(feature_belongs_to_cluster, representative)
        print(f'{Constant.line}\nComputing recall for k={i}\n{Constant.line}')
        recall = cluster.compute_recall(feature_belongs_to_cluster, representative)
        print(f'{Constant.line}\nComputing f-score for k={i}\n{Constant.line}')
        f_score = cluster.compute_f_score(precision, recall)