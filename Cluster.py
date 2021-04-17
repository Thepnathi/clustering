from math import sqrt
from sys import maxsize
from random import randrange
from Dataset_Handler import Dataset_Handler

class Cluster():
    def k_means(self, feature_dataset, k: int):
        k_representative = [] # Contains the k representative or points as the centroid of clusters
        for _ in range(k):
            random_representative = feature_dataset[randrange(len(feature_dataset))]  
            k_representative.append(random_representative) # What happens if we have the same representative?
        
        feature_centroid_index = []     # Stores the index of representative that match the index feature dataset
        for _, row in enumerate(feature_dataset):
            min_dist, closest_rep = maxsize, None
            for j in k_representative:
                distance = self.euclidean_distance(k_representative[j], row)
                if distance < min_dist:
                    min_dist = distance
                    closest_rep = j
            feature_centroid_index.append(closest_rep)
        
        

        
    def euclidean_distance(self, vec1, vec2):
        total = 0
        for i in range(len(vec1)):
            total += pow(vec1[i]- vec2[i], 2)
        return sqrt(total)
        

    def compute_precision(self):
        pass

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

    cluster.k_means(feature, 5)