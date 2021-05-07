import numpy as np 

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

from sklearn import preprocessing
import math

if __name__ == "__main__":
    data_handler = Dataset_Handler()
    # a, b, c = data_handler.load_dataset('animals')

    data_handler.load_multiple_dataset("animals", "countries", "fruits", "veggies")

    cat = data_handler.get_label_categories()
    label = data_handler.get_label_dataset()
    feature = data_handler.get_feature_dataset()
    feature_normalised = data_handler.get_feature_normalised_dataset()

    # Test l2 unit vector - all object should output 1 after normalised
    print(data_handler.test_l2_unit_vectors(feature_normalised))