import pandas as pd
import matplotlib.pyplot as plt

data_test_1 = pd.read_csv('Assig1-Dataset/test_with_label_1.csv')
data_test_2 = pd.read_csv('Assig1-Dataset/test_with_label_2.csv')
data_train_1 = pd.read_csv('Assig1-Dataset/train_1.csv')
data_train_2 = pd.read_csv('Assig1-Dataset/train_2.csv')
data_val_1 = pd.read_csv('Assig1-Dataset/val_1.csv')
data_val_2 = pd.read_csv('Assig1-Dataset/val_2.csv')

data_test_1.iloc[:, -1].value_counts().sort_index()\
    .plot(kind='bar', title='Distribution of Letters by Index in test_with_label_1')
fig = plt.gcf()
fig.savefig('Output/Initial_Distributions/distribution_test1.pdf')
plt.show()
data_test_2.iloc[:, -1].value_counts().sort_index()\
    .plot(kind='bar', title='Distribution of Letters by Index in test_with_label_2')
fig = plt.gcf()
fig.savefig('Output/Initial_Distributions/distribution_test2.pdf')
plt.show()
data_train_1.iloc[:, -1].value_counts().sort_index()\
    .plot(kind='bar', title='Distribution of Letters by Index in train_1')
fig = plt.gcf()
fig.savefig('Output/Initial_Distributions/distribution_train1.pdf')
plt.show()
data_train_2.iloc[:, -1].value_counts().sort_index()\
    .plot(kind='bar', title='Distribution of Letters by Index in train_2')
fig = plt.gcf()
fig.savefig('Output/Initial_Distributions/distribution_train2.pdf')
plt.show()
data_val_1.iloc[:, -1].value_counts().sort_index()\
    .plot(kind='bar', title='Distribution of Letters by Index in val_1')
fig = plt.gcf()
fig.savefig('Output/Initial_Distributions/distribution_val1.pdf')
plt.show()
data_val_2.iloc[:, -1].value_counts().sort_index()\
    .plot(kind='bar', title='Distribution of Letters by Index in val_2')
fig = plt.gcf()
fig.savefig('Output/Initial_Distributions/distribution_val2.pdf')
plt.show()






