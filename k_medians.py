from cluster import Cluster
from dataset_handler import Dataset_Handler
from constant import Constant
from sys import maxsize

class K_Medians(Cluster):
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
                if total_objects_in_cluster % 2 == 0:   # Even number of values, we calculate the mod between two values of middle
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
    dataset_handler = Dataset_Handler()
    dataset_handler.load_multiple_dataset("animals", "countries", "fruits", "veggies")

    category = dataset_handler.get_label_categories()
    label_dataset = dataset_handler.get_label_dataset()
    feature_dataset = dataset_handler.get_feature_dataset()
    feature_normalised_dataset = dataset_handler.get_feature_normalised_dataset()

    cluster = K_Medians(label_dataset, feature_dataset, category)

    for i in range(1, 10):
        feature_belongs_to_cluster, cluster_representative = cluster.k_medians(i)
        cluster.compute_B_CUBED(feature_belongs_to_cluster, cluster_representative, i)

    print(f'{Constant.line}\nNormalised version\n{Constant.line}')
    normalised_cluster = K_Medians(label_dataset, feature_normalised_dataset, category)
    for i in range(1, 10):
        feature_belongs_to_cluster, cluster_representative = normalised_cluster.k_medians(i)
        cluster.compute_B_CUBED(feature_belongs_to_cluster, cluster_representative, i)


