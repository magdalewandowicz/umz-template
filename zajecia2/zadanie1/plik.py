import pandas as pd
import seaborn as sns
import os
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

rtrain = pd.read_csv('train/train.tsv', sep='\t', names=["Occupancy", "date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])

print(rtrain.describe())
print('-' * 100)
print('Occupancy % :', end='')
print(sum(rtrain.Occupancy) / len(rtrain))
print('zero rule model accuracy on training set is',1 - sum(rtrain.Occupancy) / len(rtrain))
print('-' * 100)

lr_full = LogisticRegression()
X = pd.DataFrame(rtrain, columns=['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio'])
lr_full.fit(X, rtrain.Occupancy)

print('True Positives: ', end ='')
TP1=sum((lr_full.predict(X) == rtrain.Occupancy) & (lr_full.predict(X) == 1))
print(TP1)
print('True Negatives: ', end ='')
TN1=sum((lr_full.predict(X) == rtrain.Occupancy) & (lr_full.predict(X) == 0))
print(TN1)
print('False Positives: ', end ='')
FP1=sum((lr_full.predict(X) != rtrain.Occupancy) & (lr_full.predict(X) == 1))
print(FP1)
print('False Negatives: ', end ='')
FN1=sum((lr_full.predict(X) != rtrain.Occupancy) & (lr_full.predict(X) == 0))
print(FN1)
print('-'*100)

print('lr model on all variables except date accuracy on training data: ', end='')
print(sum(lr_full.predict(X) == rtrain.Occupancy) / len(rtrain))
print('sensitivity on training data:', end = '')
print(TP1/(TP1+FN1))
print('specificity on training data:', end = '')
print(TN1/(TN1+FP1))


rdev = pd.read_csv('dev-0/in.tsv', sep='\t', names=["date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])
rdev = pd.DataFrame(rdev,columns = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])
rdev2 = pd.read_csv('test-A/in.tsv', sep='\t', names=["date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])
rdev2 = pd.DataFrame(rdev2,columns = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])
rdev_expected = pd.read_csv('dev-0/expected.tsv', sep='\t', names=['y'])
print('-'*100)

print('Occupancy % :', end='')
print(sum(rdev_expected.y) / len(rdev_expected))
print('zero rule model accuracy on dev set is',1 - sum(rdev_expected.y) / len(rdev_expected))
print('-' * 100)

print('True Positives: ', end ='')
TP2=sum((lr_full.predict(rdev) == rdev_expected['y']) & (lr_full.predict(rdev) == 1))
print(TP2)
print('True Negatives: ', end ='')
TN2=sum((lr_full.predict(rdev) == rdev_expected['y']) & (lr_full.predict(rdev) == 0))
print(TN2)
print('False Positives: ', end ='')
FP2=sum((lr_full.predict(rdev) != rdev_expected['y']) & (lr_full.predict(rdev) == 1))
print(FP2)
print('False Negatives: ', end ='')
FN2=sum((lr_full.predict(rdev) != rdev_expected['y']) & (lr_full.predict(rdev) == 0))
print(FN2)
print('-'*100)

print('accuracy on dev data (full model):', end = '')
print(sum(lr_full.predict(rdev) == rdev_expected['y'] ) / len(rdev_expected))
print('sensitivity on dev data (full model):', end = '')
print(TP2/(TP2+FN2))
print('specificity on dev data (full model):', end = '')
print(TN2/(TN2+FP2))


print('-'*100)
print('writing to the expected file')
f = open('dev-0/out.tsv', 'w')
for i in range(0, len(lr_full.predict(rdev))):
    f.write(str(lr_full.predict(rdev)[i]))
    f.write('\n')
f.close()

g = open('test-A/out.tsv', 'w')
for i in range(0, len(lr_full.predict(rdev2))):
    g.write(str(lr_full.predict(rdev2)[i]))
    g.write('\n')
g.close()


print('-'*100)

sns.regplot(x=rdev.CO2, y=rdev_expected.y, logistic=True, y_jitter=.1)
plt.show()

