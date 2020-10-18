import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
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

classifier = DecisionTreeClassifier()
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, None],
    'min_samples_split': [*range(2, 4, 1)],
    'min_impurity_decrease': [*range(0, 4, 1)],
    'class_weight': [None, 'balanced'],
}
clf = GridSearchCV(classifier, param_grid, verbose=2, n_jobs=-1)
clf.fit(train_features, train_target)

valid_prediction = clf.predict(valid_features)

valid_confusion = confusion_matrix(valid_target, valid_prediction)
p1, r1, f1, _ = precision_recall_fscore_support(valid_target, valid_prediction)
p2, r2, f2, _ = precision_recall_fscore_support(valid_target, valid_prediction, average='weighted')
p3, r3, f3, _ = precision_recall_fscore_support(valid_target, valid_prediction, average='macro')
valid_accuracy = accuracy_score(valid_target, valid_prediction)

file = open('Output/Best-DT-DS2.csv', 'w', encoding='utf8')
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


