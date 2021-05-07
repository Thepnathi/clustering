from cluster import Cluster_Algorithm
from dataset_handler import Dataset_Handler
from constant import Constant
from plot import Plot_B_CUBED
from sys import maxsize

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
                    if total_objects_in_cluster % 2 == 0:   # Even number of values, we calculate the mod between two values of middle
                        # print(f'Length of current dimension is {len(median_incrementer[m])}')
                        # print(f'Compute the lower bound is {(total_objects_in_cluster//2)-1}')
                        # print(f'Compute the upper bound is {total_objects_in_cluster//2}')
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
    data_handler = Dataset_Handler()
    plot_tool = Plot_B_CUBED()
    data_handler.load_multiple_dataset("animals", "countries", "fruits", "veggies") # Load the four category data files and merge into one

    category = data_handler.get_label_categories()
    label_dataset = data_handler.get_label_dataset()
    feature_dataset = data_handler.get_feature_dataset()
    feature_normalised_dataset = data_handler.get_feature_normalised_dataset()

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