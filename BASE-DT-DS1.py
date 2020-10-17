import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import csv

data_train = pd.read_csv('Assig1-Dataset/train_1.csv')
data_test = pd.read_csv('Assig1-Dataset/test_with_label_1.csv')
data_valid = pd.read_csv('Assig1-Dataset/val_1.csv')

train = data_train.copy()
test = data_test.copy()
valid = data_valid.copy()

train_features = train.loc[:, '1':'1.827']
train_target = train['1.828']

test_features = test.loc[:, '1':'1.849']
test_target = test['4']

valid_features = valid.loc[:, '1':'1.842']
valid_target = valid['1.843']

classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(train_features, train_target)

test_prediction = classifier.predict(test_features)
# print(accuracy_score(test_prediction, test_target) * 100)


valid_prediction = classifier.predict(valid_features)

valid_confusion = confusion_matrix(valid_target, valid_prediction)
p1, r1, f1, s = precision_recall_fscore_support(valid_target, valid_prediction)
p2, r2, f2, s = precision_recall_fscore_support(valid_target, valid_prediction, average='weighted')
p3, r3, f3, s = precision_recall_fscore_support(valid_target, valid_prediction, average='macro')
valid_accuracy = accuracy_score(valid_target, valid_prediction)

print(valid_accuracy)
print(p1)
print(r1)
print(f1)


file = open('Output/Base-DT-DS1.csv', 'w', encoding='utf8')
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
