from math import sqrt
from sys import maxsize
from random import randrange
from Dataset_Handler import Dataset_Handler

class Cluster():
    def k_means(self, feature_dataset, k: int):
        # Step 1: Initialisation phase
        representative = [] # Contains the k representative of clusters or points as the centroid of clusters
        for _ in range(k):
            random_representative = feature_dataset[randrange(len(feature_dataset))]  
            representative.append(random_representative) # What happens if we have the same representative?
        # Repeat until convergence.
        # This means no objects have moved among clusters
        # or by number of iteration specified
        convergence_steps = 0
        feature_belongs_to_cluster = []
        while True:
            convergence_steps += 1
            # Step 2: Assignment phase
            # Assign each feature points to a cluster or representative
            feature_belongs_to_cluster_temp = []     # Stores the index of representative that match the index feature dataset
            for _, row in enumerate(feature_dataset):
                min_dist, closest_rep = maxsize, None
                for j in range(len(representative)):
                    distance = self.euclidean_distance(representative[j], row)
                    if distance < min_dist:
                        min_dist = distance
                        closest_rep = j
                feature_belongs_to_cluster_temp.append(closest_rep)
            # Step 3: Optimisation phase
            # Calculate new representative - mean of all item within the cluste
            representative = self.compute_new_representative(representative, feature_belongs_to_cluster_temp, feature_dataset)
            if feature_belongs_to_cluster_temp == feature_belongs_to_cluster:
                    break
            feature_belongs_to_cluster = feature_belongs_to_cluster_temp
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

    def euclidean_distance(self, vec1, vec2):
        total = 0
        for i in range(len(vec1)):
            total += pow(vec1[i]- vec2[i], 2)
        return sqrt(total)
        

    def compute_precision(self, feature_dataset, label_dataset, category, feature_belongs_to_cluster, representative):
        # for each cluster we will calculate tje dominant label
        for i in range(len(representative)):
            category_count = {label[0]: 0 for label in category}
            # Iterate through the whole feature dataset with the label to indicate which cluster it belongs to 
            for j in range(len(feature_belongs_to_cluster)):
                if feature_belongs_to_cluster[j] == i: # This feature belongs to the current cluster
                    # Now we want to find out what it is 
                    for label in category:
                        if label_dataset[j] in label[1]:
                            category_count[label[0]] += 1
            print(category_count)

    def compute_recall(self):
        pass

    def compute_f_score(self):
        pass

    def compute_B_CUBED(self):
        pass


if __name__ == "__main__":
    data_handler = Dataset_Handler()
    cluster = Cluster()
    data_handler.load_multiple_dataset("animals", "countries", "fruits", "veggies")
    cat = data_handler.get_label_categories()
    label, feature = data_handler.get_label_dataset(), data_handler.get_feature_dataset()

    feature_belongs_to_cluster, representative = cluster.k_means(feature, 4)
    cluster.compute_precision(feature, label, cat, feature_belongs_to_cluster, representative)
