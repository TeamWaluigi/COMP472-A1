import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import csv

data_train = pd.read_csv('Assig1-Dataset/train_1.csv')
data_test = pd.read_csv('Assig1-Dataset/test_with_label_1.csv')
data_valid = pd.read_csv('Assig1-Dataset/val_1.csv')

# TODO I made these dynamic, but the BASE-DT and BEST-DT results have changed slightly! Check into this later
train_features = data_train.iloc[:, :-1]
train_target = data_train.iloc[:, -1]
test_features = data_test.iloc[:, :-1]
test_target = data_test.iloc[:, -1]
valid_features = data_valid.iloc[:, :-1]
valid_target = data_valid.iloc[:, -1]

classifier = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=7, min_impurity_decrease=0, class_weight='balanced')
# See excel file BEST-DT-DS1 to see comparisons of different results. This was chosen to be the best assuming we favour recall over accura
classifier.fit(train_features, train_target)

test_prediction = classifier.predict(test_features)
# print(accuracy_score(test_prediction, test_target) * 100)


valid_prediction = classifier.predict(valid_features)

valid_confusion = confusion_matrix(valid_target, valid_prediction)
p1, r1, f1, s = precision_recall_fscore_support(valid_target, valid_prediction, zero_division='warn')
p2, r2, f2, s = precision_recall_fscore_support(valid_target, valid_prediction, average='weighted')
p3, r3, f3, s = precision_recall_fscore_support(valid_target, valid_prediction, average='macro')
valid_accuracy = accuracy_score(valid_target, valid_prediction)

print(valid_accuracy)
print(p1)
print(r1)
print(f1)

file = open('Output/Best-DT-DS1.csv', 'w', encoding='utf8')
writer = csv.writer(file, quotechar='"', quoting=csv.QUOTE_ALL, lineterminator='\n')
count = 0
for x in valid_prediction:
    writer.writerow([count, x])
    count += 1
writer.writerow("")
writer.writerow("")
writer.writerow(["Confusion Matrix"])
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
           'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
writer.writerow(letters)
for x in range(25):
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
