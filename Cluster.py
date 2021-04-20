from constant import Constant
from abc import ABC, abstractmethod
from sys import maxsize
from random import randrange

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
        for _ in range(k):
            random_index = randrange(self.dataset_length) 
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
            print(object_type_count)
            print(total_objects_in_cluster)
        print(len(precision_per_object))
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
            print(object_type_count)
        print(len(recall_per_object))
        return recall_per_object

    def compute_f_score(self, precision_per_cluster, recall_per_cluster):
        f_score_per_cluster = []
        for i in range(len(precision_per_cluster)):
            precision = precision_per_cluster[i][1]
            recall = recall_per_cluster[i][1]
            f_score = (2 * precision * recall) / (precision + recall) 
            f_score_per_cluster.append(f_score)
        return f_score_per_cluster

    def compute_B_CUBED(self, feature_belongs_to_cluster, cluster_representative, k):
        # print(f'{Constant.line}\nComputing precision for k={k}\n{Constant.line}')
        precision_per_object = self.compute_precision(feature_belongs_to_cluster, cluster_representative)
        # print(f'Precision for each cluster: \n{precision}')
        # print(f'{Constant.line}\nComputing recall for k={k}\n{Constant.line}')
        recall_per_object = self.compute_recall(feature_belongs_to_cluster, cluster_representative)
        # print(f'Recall for each cluster: \n{recall}')

        # print(f'{Constant.line}\nComputing f-score for k={k}\n{Constant.line}')
        # f_score = self.compute_f_score(precision, recall)
        # print(f'F-score for each cluster: \n{f_score}')
        # print("\n")
