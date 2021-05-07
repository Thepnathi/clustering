# =========================================
# Name: Thepnathi Chindalaksanaloet
# Student ID: 201123978
# =========================================

import numpy as np 
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from sys import maxsize
from math import sqrt
from collections import namedtuple
from random import randrange
from typing import List

class Constant:
    line = "=====" * 10

# This class deals with reading and writing the data files
class Dataset_Handler:
    def __init__(self, path="CA2data"):
        self.path = path
        self.label_dataset = None
        self.feature_dataset = None
        self.feature_normalised_dataset = None
        self.label_categories = None

    def get_label_dataset(self):
        return self.label_dataset

    def get_feature_dataset(self):
        return self.feature_dataset

    def get_feature_normalised_dataset(self):
        return self.feature_normalised_dataset
                
    def get_label_categories(self):
        return self.label_categories

    # Load a single dataset and store in labels and features list
    def load_dataset(self, file_name: str):
        opened_file = open(f'{self.path}/{file_name}', 'r')
        file_lines = opened_file.readlines()

        labels, features, features_normalised = [], [], []

        for line in file_lines:
            line = line.strip("\n")
            split_line_by_delimeter = line.split(" ")  # each line will be splited by a delimeter and return as list
            labels.append(split_line_by_delimeter[0])
            float_feature = [float(item) for item in split_line_by_delimeter[1:]]
            normalised_float_feature = self.normalise_l2_vector(float_feature)
            features.append(float_feature)
            features_normalised.append(normalised_float_feature)

        return labels, features, features_normalised

    # Load multiple dataset and store the features and corresponding labels into two list
    def load_multiple_dataset(self, *file_names):
        label_categories = []
        label_dataset = []
        feature_dataset = []
        feature_normalised_dataset = []

        for name in file_names:
            labels, features, features_normalised = self.load_dataset(name)
            label_dataset += labels
            feature_dataset += features
            feature_normalised_dataset += features_normalised
            
            label_dict = {label for label in labels}
            label_categories.append((name, label_dict))

        self.label_categories = label_categories
        self.label_dataset = label_dataset
        self.feature_dataset = feature_dataset
        self.feature_normalised_dataset = feature_normalised_dataset

    def normalise_l2_vector(self, vec):
        norm = np.linalg.norm(vec, ord=2)
        return vec/norm

    def test_l2_unit_vectors(self, normalised_vectors):
        normalised_vectors = np.array(normalised_vectors)
        squared_vectors = normalised_vectors ** 2
        sum_squared_vectors = np.sum(squared_vectors, axis=1)
        return sum_squared_vectors

class Plot_B_CUBED:
    def plot_b_cubed(self, precision: List[float], recall: List[float], f_score: List[float], k: List[int], title="B-CUBED Measure",):
       plt.plot(k, precision, color="red", marker="X", label="Precision")
       plt.plot(k, recall, color="green", marker="o", label="Recall")
       plt.plot(k, f_score, color="blue", marker="^", label="F-score")
       plt.title(title, fontsize=14)
       plt.xlabel('F values', fontsize=10)
       plt.ylabel('B-CUBED precision, recall and F-score', fontsize=10)
       plt.legend(loc="upper left")
       plt.grid(True)
       plt.show()

class Cluster_Algorithm(ABC):
    def __init__(self, label_dataset, feature_dataset, category):
        self.label_dataset = label_dataset
        self.feature_dataset = feature_dataset
        self.category = category
        self.dataset_length = len(self.feature_dataset)

    @abstractmethod
    def compute_new_cluster_representatives(self, cluster_representative, feature_belongs_to_cluster):
        pass

    # Pick k random data or features from the dataset
    # Improve by not select same data point
    def compute_random_cluster_representative(self, k: int):
        cluster_representative = []
        feature_index = set()
        for _ in range(k):
            random_index = randrange(self.dataset_length) 
            if random_index in feature_index:       
                while random_index in feature_index: # prevents the program from selecting the same feature as the representative
                    random_index = randrange(self.dataset_length) 
            feature_index.add(random_index)
            random_representative = self.feature_dataset[randrange(self.dataset_length)]
            cluster_representative.append(random_representative)
        return cluster_representative
    
    # For each types of objects in cluster. We compute the quantity of each type
    # We can optimise this further by removing the feature with current index, so next iteration will be less
    def compute_object_type_quantity_in_cluster(self, feature_belongs_to_cluster, representative_index):
        object_type_count = {object_type[0]: 0 for object_type in self.category}
        for i in range(len(feature_belongs_to_cluster)):
            if feature_belongs_to_cluster[i] == representative_index: # This feature belongs to the current cluster
                # Now we find out the label of this object and increment that object type by one
                for label in self.category:
                    if self.label_dataset[i] in label[1]:
                        object_type_count[label[0]] += 1
        return object_type_count

    def compute_precision(self, feature_belongs_to_cluster, cluster_representative):
        precision_per_object = []
        # Iterate through each cluster and find objects within this cluster
        for i in range(len(cluster_representative)):
            object_type_count = {object_type[0]: 0 for object_type in self.category}
            # Iterate through the list with the value that indicate which cluster it belongs to
            for j in range(len(feature_belongs_to_cluster)):
                if feature_belongs_to_cluster[j] == i: # This feature belongs to the current cluster rep
                    # Now we find out what the object type is and increment the count
                    for label in self.category:
                        if self.label_dataset[j] in label[1]:  # Label dataset share same index as the list
                            object_type_count[label[0]] += 1
            # Now we calculate the precision for every object in this cluster and append to list
            total_objects_in_cluster = sum([object_type_count[label] for label in object_type_count])
            for label in object_type_count:
                objects_precision = [object_type_count[label]/total_objects_in_cluster] * object_type_count[label]
                precision_per_object += objects_precision
        return precision_per_object

    def compute_recall(self, feature_belongs_to_cluster, cluster_representative):
        recall_per_object = []
        # Iterate through each cluster and find objects within this cluster
        # Follows through the same principle as precision except the last part
        for i in range(len(cluster_representative)):
            object_type_count = {object_type[0]: 0 for object_type in self.category}
            for j in range(len(feature_belongs_to_cluster)):
                if feature_belongs_to_cluster[j] == i:
                    for label in self.category:
                        if self.label_dataset[j] in label[1]:
                            object_type_count[label[0]] += 1
            dataset_count_by_object = {object_type[0]:len(object_type[1]) for object_type in self.category}
            for label in object_type_count:
                object_recall = [object_type_count[label]/dataset_count_by_object[label]] * object_type_count[label]
                recall_per_object += object_recall
        return recall_per_object

    def compute_f_score(self, precision_per_object, recall_per_object):
        f_score_per_object = []
        for i in range(len(precision_per_object)):
            precision = precision_per_object[i]
            recall = recall_per_object[i]
            f_score = (2 * precision * recall) / (precision + recall)
            f_score_per_object.append(f_score)
        return f_score_per_object

    def compute_B_CUBED(self, feature_belongs_to_cluster, cluster_representative):
        B_CUBED = namedtuple('B_CUBED', 'precision recall f_score')
        total_objects_in_cluster = len(feature_belongs_to_cluster)
        precision_per_object = self.compute_precision(feature_belongs_to_cluster, cluster_representative)
        recall_per_object = self.compute_recall(feature_belongs_to_cluster, cluster_representative)
        f_score_per_object = self.compute_f_score(precision_per_object, recall_per_object)  

        average_precision = sum(precision_per_object) / total_objects_in_cluster
        average_recall = sum(recall_per_object) / total_objects_in_cluster
        average_f_score = sum(f_score_per_object) / total_objects_in_cluster
        
        result = B_CUBED(precision=average_precision, recall=average_recall, f_score=average_f_score)
        return result
    

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

class K_Medians_Algorithm(Cluster_Algorithm):
    def __init__(self, label_dataset, feature_dataset, category):
        super().__init__(label_dataset, feature_dataset, category)

    def k_medians(self, k: int):
        # Step 1 - Initialisation phase
        cluster_representative = self.compute_random_cluster_representative(k)
        
        convergence_steps = 0
        feature_belongs_to_cluster = []
        while True:    # Repeat step 2 and 3 until convergence (no object has moved between clusters or by specified iteration)
            convergence_steps += 1
            feature_belongs_to_cluster_temp = [] 
            # Step 2 - Assignment phase. Assign each feature or data object to closest cluster representative
            for _, row in enumerate(self.feature_dataset): 
                min_dist, closest_cluster_rep = maxsize, None
                # Iterate through each cluster representative so we can find the closest to current object
                for ith_cluster in range(len(cluster_representative)):
                    distance = self.manhattan_distance(cluster_representative[ith_cluster], row)
                    if distance < min_dist: # Check if the current cluster rep is closer than previous rep
                        min_dist, closest_cluster_rep = distance, ith_cluster
                feature_belongs_to_cluster_temp.append(closest_cluster_rep) # We append the index of the cluster for this feature index
            # Step 3 - Optimisation phase. Calculate new cluster representative - median point in each clusters
            cluster_representative = self.compute_new_cluster_representatives(cluster_representative, feature_belongs_to_cluster_temp)
            if feature_belongs_to_cluster_temp == feature_belongs_to_cluster:
                break # Break the While loop once the object no moved between clusters
            feature_belongs_to_cluster = feature_belongs_to_cluster_temp
        print(f"k-medians algorithm with k={k} cluster took {convergence_steps} steps until converged.")
        return feature_belongs_to_cluster, cluster_representative

    def compute_new_cluster_representatives(self, cluster_representative, feature_belongs_to_cluster):
        for i in range(len(cluster_representative)):
            median_incrementer = [[-maxsize] for _ in range(len(self.feature_dataset[0]))]
            total_objects_in_cluster = 0
            for j in range(len(feature_belongs_to_cluster)):
                if feature_belongs_to_cluster[j] == i:
                    total_objects_in_cluster += 1
                    for d in range(len(self.feature_dataset[j])): # Iterate through the feature and increment each dimension 
                        median_incrementer[d].append(self.feature_dataset[j][d])

            for m in range(len(median_incrementer)):   # Now we compute the median for each dimension
                median_incrementer[m] = sorted(median_incrementer[m][1:]) # We sort all the values and take the median
                dimension_len = len(median_incrementer[m])
                if dimension_len != 0:
                    if total_objects_in_cluster % 2 == 0:   # Even number of values, we calculate the median between two values of middle
                        lower_mid = median_incrementer[m][(total_objects_in_cluster//2)-1]
                        upper_mid = median_incrementer[m][total_objects_in_cluster//2]
                        median = (lower_mid + upper_mid) / 2
                        cluster_representative[i][m] = median
                    else:
                        median = median_incrementer[m][(total_objects_in_cluster//2)]
                        cluster_representative[i][m] = median
        return cluster_representative

    def manhattan_distance(self, vec1, vec2):
        total = 0
        for i in range(len(vec1)):
            total += abs(vec1[i] - vec2[i])
        return total

if __name__ == "__main__":
    # Initialise the data handler and the matplotlib tool
    data_handler = Dataset_Handler()
    plot_tool = Plot_B_CUBED()
    data_handler.load_multiple_dataset("animals", "countries", "fruits", "veggies") # Load the four category data files and merge into one

    # Get and store all the required dataset for the cluster algorithm
    category = data_handler.get_label_categories() # category dictionary for constant lookups i.e. [['animals', ('elephant', 'leopard'....)], ['countries, (...)]...] 
    label_dataset = data_handler.get_label_dataset() # label corresponds to the index of the feature dataset 
    feature_dataset = data_handler.get_feature_dataset() 
    feature_normalised_dataset = data_handler.get_feature_normalised_dataset()

    # Uncomment this to test l2 unit vector - all object should output 1 after normalised
    # print(data_handler.test_l2_unit_vectors(feature_normalised))

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

    print(f'\n{Constant.line}\nNormal dataset version\n{Constant.line}\n')

    cluster = K_Medians_Algorithm(label_dataset, feature_dataset, category)
    precision_per_k, recall_per_k, f_score_per_k, k_values = [], [], [], []
    for i in range(1, 10):
        feature_belongs_to_cluster, cluster_representative = cluster.k_medians(i)
        result = cluster.compute_B_CUBED(feature_belongs_to_cluster, cluster_representative)
        print(result)
        k_values.append(i)
        precision_per_k.append(result.precision)
        recall_per_k.append(result.recall)
        f_score_per_k.append(result.f_score)
    plot_tool.plot_b_cubed(precision_per_k, recall_per_k, f_score_per_k, k_values, "K-Medians Cluster B-CUBED Measure")

    print(f'\n{Constant.line}\nNormalised L2 dataset version\n{Constant.line}\n')

    cluster = K_Medians_Algorithm(label_dataset, feature_normalised_dataset, category)
    precision_per_k, recall_per_k, f_score_per_k, k_values = [], [], [], []
    for i in range(1, 10):
        feature_belongs_to_cluster, cluster_representative = cluster.k_medians(i)
        result = cluster.compute_B_CUBED(feature_belongs_to_cluster, cluster_representative)
        print(result)
        k_values.append(i)
        precision_per_k.append(result.precision)
        recall_per_k.append(result.recall)
        f_score_per_k.append(result.f_score)
    plot_tool.plot_b_cubed(precision_per_k, recall_per_k, f_score_per_k, k_values, "K-Medians Cluster B-CUBED Measure L2 Norm")