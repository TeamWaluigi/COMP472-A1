import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import csv

output_results_file_path = 'Output/Base-DT-DS1.csv'
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
           'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Read our data sets
data_train = pd.read_csv('Assig1-Dataset/train_1.csv')
data_valid = pd.read_csv('Assig1-Dataset/val_1.csv')
data_test = pd.read_csv('Assig1-Dataset/test_with_label_1.csv')

# We'll combine our training and validation data since we won't be tweaking our hyper parameters yet
data_valid.columns = data_train.columns
combined_training_set = pd.concat([data_train, data_valid], ignore_index=True, axis=0)
testing_set = data_test

# Define the features and target sets
combined_training_set_features = combined_training_set.iloc[:, :-1]
combined_training_set_target = combined_training_set.iloc[:, -1]
testing_set_features = data_test.iloc[:, :-1]
testing_set_target = data_test.iloc[:, -1]

# Define the classifier used
classifier = DecisionTreeClassifier(criterion='entropy')

# Fit accordingly to our combined training and validation set
classifier.fit(combined_training_set_features, combined_training_set_target)

# Finally, predict the test set the best estimator that GridSearch has scored during training
test_prediction = classifier.predict(testing_set_features)

# Metrics
test_confusion = confusion_matrix(testing_set_target, test_prediction)
p1, r1, f1, _ = precision_recall_fscore_support(testing_set_target, test_prediction, zero_division='warn')
p2, r2, f2, _ = precision_recall_fscore_support(testing_set_target, test_prediction, average='weighted')
p3, r3, f3, _ = precision_recall_fscore_support(testing_set_target, test_prediction, average='macro')
test_accuracy = accuracy_score(testing_set_target, test_prediction)

# Write results and metrics to file
file = open(output_results_file_path, 'w', encoding='utf8')
writer = csv.writer(file, quotechar='"', quoting=csv.QUOTE_ALL, lineterminator='\n')
count = 0
for x in test_prediction:
    writer.writerow([count, x])
    count += 1
writer.writerow("")
writer.writerow("")
writer.writerow(["Confusion Matrix"])
writer.writerow(letters)
for x in range(len(letters)):
    writer.writerow(test_confusion[x])
writer.writerow("")
writer.writerow("")
writer.writerow(["Precision"])
writer.writerow(letters)
writer.writerow(p1)
writer.writerow(["Recall"])
writer.writerow(letters)
writer.writerow(r1)
writer.writerow("")
writer.writerow(["F1-Measures"])
writer.writerow(letters)
writer.writerow(f1)
writer.writerow("")
writer.writerow(['Accuracy', test_accuracy])
writer.writerow(['F1-Macro', f3])
writer.writerow(['F1-Weighted', f2])

