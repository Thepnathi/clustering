from math import sqrt
from sys import maxsize
from random import randrange
from Dataset_Handler import Dataset_Handler

class Cluster():
    def __init__(self, label_dataset, feature_dataset, category):
        self.label_dataset = label_dataset
        self.feature_dataset = feature_dataset
        self.category = category

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
        
    # It is the number the same catgory or type of object over all the object in the cluster
    def compute_precision(self, feature_belongs_to_cluster, representative):
        result = []
        # for each cluster we will calculate the dominant label
        for i in range(len(representative)):
            object_type_count = {object_type[0]: 0 for object_type in self.category}
            # Iterate through the whole feature dataset with the label to indicate which cluster it belongs to 
            for j in range(len(feature_belongs_to_cluster)):
                if feature_belongs_to_cluster[j] == i: # This feature belongs to the current cluster
                    # Now we want to find out what it is 
                    for label in self.category:
                        if label_dataset[j] in label[1]:
                            object_type_count[label[0]] += 1
            # Iterate through all the objects in current cluster and find object type that appeared most and divide by total objects in cluster
            total_objects_in_cluster = 0
            dominant_object_type = ["label", -1]
            for category in object_type_count:
                total_objects_in_cluster += object_type_count[category]
                dominant_object_type = [category, object_type_count[category]] if object_type_count[category] > dominant_object_type[1] else dominant_object_type
            dominant_object_type[1] = dominant_object_type[1] / total_objects_in_cluster
            result.append(dominant_object_type)
            print(object_type_count)   
        return result


    # It is the number of instance of an object in a cluster over all of the dataset
    def compute_recall(self, feature_belongs_to_cluster, representative):
        result = []
        for i in range(len(representative)):
            object_type_count = {object_type[0]: 0 for object_type in self.category}
            for j in range(len(feature_belongs_to_cluster)):
                if feature_belongs_to_cluster[j] == i:
                    for label in self.category:
                        if label_dataset[j] in label[1]:
                            object_type_count[label[0]] += 1
            # Iterate through all the objects in current cluster and find object type that appeared most divide by all that object type in the whole dataset
            dominant_object_type = ["label", -1]
            for category in object_type_count:
                dominant_object_type = [category, object_type_count[category]] if object_type_count[category] > dominant_object_type[1] else dominant_object_type
            
            total_object_type_in_dataset = 0
            for object_type in self.category:
                if object_type[0] == dominant_object_type[0]:
                    total_object_type_in_dataset += len(object_type[1])
            dominant_object_type[1] = dominant_object_type[1] / total_object_type_in_dataset
            result.append(dominant_object_type)
            print(object_type_count)
        return result

    def compute_f_score(self):
        pass

    def compute_B_CUBED(self):
        pass


if __name__ == "__main__":
    data_handler = Dataset_Handler()
    data_handler.load_multiple_dataset("animals", "countries", "fruits", "veggies") # Load the four category data files and merge into one

    category = data_handler.get_label_categories()
    label_dataset, feature_dataset = data_handler.get_label_dataset(), data_handler.get_feature_dataset()

    cluster = Cluster(label_dataset, feature_dataset, category)

    feature_belongs_to_cluster, representative = cluster.k_means(feature_dataset, 4)
    # precision = cluster.compute_precision(feature_belongs_to_cluster, representative)
    recall = cluster.compute_recall(feature_belongs_to_cluster, representative)
    print(recall)
