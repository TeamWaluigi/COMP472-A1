import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import csv

data_train = pd.read_csv('Assig1-Dataset/train_2.csv')
data_test = pd.read_csv('Assig1-Dataset/test_with_label_2.csv')
data_valid = pd.read_csv('Assig1-Dataset/val_2.csv')

letters = ['pi', 'alpha', 'beta', 'sigma', 'gamma', 'delta', 'lambda', 'omega', 'mu', 'xi']

train_features = data_train.iloc[:, :-1]
train_target = data_train.iloc[:, -1]
test_features = data_test.iloc[:, :-1]
test_target = data_test.iloc[:, -1]
valid_features = data_valid.iloc[:, :-1]
valid_target = data_valid.iloc[:, -1]

classifier = MLPClassifier()
param_grid = {
    'activation': ['logistic', 'tanh', 'relu', 'identity'],
    'hidden_layer_sizes': [(20, 20), (10, 10, 10, 10)],
    'solver': ['adam', 'sgd']
}
clf = GridSearchCV(classifier, param_grid, verbose=2, n_jobs=-1)
clf.fit(train_features, train_target)

valid_prediction = clf.predict(valid_features)

valid_confusion = confusion_matrix(valid_target, valid_prediction)
p1, r1, f1, _ = precision_recall_fscore_support(valid_target, valid_prediction)
p2, r2, f2, _ = precision_recall_fscore_support(valid_target, valid_prediction, average='weighted')
p3, r3, f3, _ = precision_recall_fscore_support(valid_target, valid_prediction, average='macro')
valid_accuracy = accuracy_score(valid_target, valid_prediction)

file = open('Output/Best-MLP-DS2.csv', 'w', encoding='utf8')
writer = csv.writer(file, quotechar='"', quoting=csv.QUOTE_ALL, lineterminator='\n')
count = 0
for x in valid_prediction:
    writer.writerow([count, x])
    count += 1
writer.writerow("")
writer.writerow("")
writer.writerow(["Confusion Matrix"])
writer.writerow(letters)
for x in range(letters.__len__()):
    writer.writerow(valid_confusion[x])
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
writer.writerow(['Accuracy', valid_accuracy])
writer.writerow(['F1-Macro', f3])
writer.writerow(['F1-Weighted', f2])
writer.writerow("")
writer.writerow(['Param grid tested:'])
writer.writerow([param_grid])
writer.writerow(['Chosen best params:'])
writer.writerow([clf.best_params_])


