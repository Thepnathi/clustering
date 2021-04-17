class Dataset_Handler:
    def __init__(self, path="CA2data"):
        self.path = path
        self.label_dataset = None
        self.feature_dataset = None
        self.label_categories = None

    def get_label_dataset(self):
        return self.label_dataset

    def get_feature_dataset(self):
        return self.feature_dataset
                
    def get_label_categories(self):
        return self.label_categories

    # Load a single dataset and store in labels and features list
    def load_dataset(self, file_name: str):
        opened_file = open(f'{self.path}/{file_name}', 'r')
        file_lines = opened_file.readlines()

        labels, features = [], []

        for line in file_lines:
            line = line.strip("\n")
            split_line_by_delimeter = line.split(" ")  # each line will be splited by a delimeter and return as list
            labels.append(split_line_by_delimeter[0])
            float_feature = [float(item) for item in split_line_by_delimeter[1:]]
            features.append(float_feature)

        return labels, features

    # Load multiple dataset and store the features and corresponding labels into two list
    def load_multiple_dataset(self, *file_names):
        label_categories = []
        label_dataset, feature_dataset = [], []

        for name in file_names:
            labels, features = self.load_dataset(name)
            label_dataset += labels
            feature_dataset += features
            
            label_dict = {label for label in labels}
            label_categories.append((name, label_dict))

        self.label_categories = label_categories
        self.label_dataset, self.feature_dataset = label_dataset, feature_dataset

if __name__ == "__main__":
    data_handler = Dataset_Handler()

    a, b = data_handler.load_dataset('animals')

    data_handler.load_multiple_dataset("animals", "countries", "fruits", "veggies")

    cat = data_handler.get_label_categories()
    label, feature = data_handler.get_label_dataset(), data_handler.get_feature_dataset()
    
    for i in range(len(label)):
        print(f'{label[i]} - {feature[i]}')