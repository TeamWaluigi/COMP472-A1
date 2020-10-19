import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import csv

from sklearn.neural_network import MLPClassifier

output_results_file_path = 'Output/Best-MLP-DS2.csv'
letters = ['pi', 'alpha', 'beta', 'sigma', 'gamma', 'delta', 'lambda', 'omega', 'mu', 'xi']

# Read our data sets
data_train = pd.read_csv('Assig1-Dataset/train_2.csv')
data_valid = pd.read_csv('Assig1-Dataset/val_2.csv')
data_test = pd.read_csv('Assig1-Dataset/test_with_label_2.csv')

# For Skylearn's GridSearchCV, we need to combine our training and validation data.
# We then define the split the GridSearchCV will use to cross-verify to be between our training and validation data
data_valid.columns = data_train.columns
gridsearch_training_set = pd.concat([data_train, data_valid], ignore_index=True, axis=0)
testing_set = data_test
ps = PredefinedSplit([-1 if entry in data_train.index else 0 for entry in gridsearch_training_set.index])
# Define the features and target sets
gridsearch_training_set_features = gridsearch_training_set.iloc[:, :-1]
gridsearch_training_set_target = gridsearch_training_set.iloc[:, -1]
testing_set_features = data_test.iloc[:, :-1]
testing_set_target = data_test.iloc[:, -1]

# Define the classifier used
classifier = MLPClassifier()

# Define the parameters that we want to be tweaked with the GridSearch
param_grid = {
    'activation': ['logistic', 'tanh', 'relu', 'identity'],
    'hidden_layer_sizes': [(50), (25, 25)],
    'solver': ['adam', 'sgd']
}

# Fit accordingly to our training and validation set (recall it has been split)
clf = GridSearchCV(classifier, param_grid, verbose=2, n_jobs=-1, cv=ps)
clf.fit(gridsearch_training_set_features, gridsearch_training_set_target)

# Finally, predict the test set the best estimator that GridSearch has scored during training
test_prediction = clf.predict(testing_set_features)

# Metrics
test_confusion = confusion_matrix(testing_set_target, test_prediction)
p1, r1, f1, _ = precision_recall_fscore_support(testing_set_target, test_prediction, zero_division='warn')
p2, r2, f2, _ = precision_recall_fscore_support(testing_set_target, test_prediction, average='weighted')
p3, r3, f3, _ = precision_recall_fscore_support(testing_set_target, test_prediction, average='macro')
test_accuracy = accuracy_score(testing_set_target, test_prediction)

# Write results and metrics to file
file = open(output_results_file_path, 'w', encoding='utf8')
writer = csv.writer(file, quotechar='"', quoting=csv.QUOTE_ALL, lineterminator='\n')
count = 1
for x in test_prediction:
    writer.writerow([count, x])
    count += 1
writer.writerow("")
writer.writerow("")
writer.writerow(["Confusion Matrix"])
letter_header = letters.copy()
letter_header.insert(0, 'Act\Pre')
writer.writerow(letter_header)
letter_index = 0
for x in range(len(letters)):
    line = test_confusion[x].tolist()
    line.insert(0, letters[letter_index])
    writer.writerow(line)
    letter_index += 1
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
writer.writerow("")
writer.writerow(['Param grid tested:'])
writer.writerow([param_grid])
writer.writerow(['Chosen best params:'])
writer.writerow([clf.best_params_])
