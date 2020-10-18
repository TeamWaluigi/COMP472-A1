import pandas as pd
import matplotlib.pyplot as plt


def print_distribution(dataset_path, dataset_name):
    dataset = pd.read_csv(dataset_path)
    dataset.iloc[:, -1].value_counts().sort_index()\
        .plot(kind='bar', title=f'Distribution of Letters by Index in {dataset_name}')
    fig = plt.gcf()
    fig.savefig(f'Output/Initial_Distributions/distribution_{dataset_name}.pdf')
    plt.show()


print_distribution('Assig1-Dataset/test_with_label_1.csv', 'Test Data Set 1')
print_distribution('Assig1-Dataset/test_with_label_2.csv', 'Test Data Set 2')
print_distribution('Assig1-Dataset/train_1.csv', 'Training Data Set 1')
print_distribution('Assig1-Dataset/train_2.csv', 'Training Data Set 2')
print_distribution('Assig1-Dataset/val_1.csv', 'Validation Data Set 1')
print_distribution('Assig1-Dataset/val_2.csv', 'Validation Test Data Set 2')
