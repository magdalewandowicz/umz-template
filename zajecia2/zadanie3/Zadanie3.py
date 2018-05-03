import pandas as pd
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

rtrain = pd.read_csv('train/in.tsv', sep='\t', names=["results", "A1", "A2", "B1", "B2", "C1", "C2","D1", "D2", "E1", "E2", "F1", "F2","G1", "G2", "H1", "H2", "I1", "I2","J1", "J2", "K1", "K2", "L1", "L2","M1", "M2", "N1", "N2", "O1", "O2","P1", "P2", "Q1", "Q2"])

print(rtrain.describe())
print('-' * 100)
print('results % :', end='')
print(sum(rtrain.results=="g") / len(rtrain))
print('zero rule model accuracy on training set is',1 - sum(rtrain.results=="g") / len(rtrain))
print('-' * 100)


lr_full = LogisticRegression()
X = pd.DataFrame(rtrain, columns=["A1", "A2", "B1", "B2", "C1", "C2","D1", "D2", "E1", "E2", "F1", "F2","G1", "G2", "H1", "H2", "I1", "I2","J1", "J2", "K1", "K2", "L1", "L2","M1", "M2", "N1", "N2", "O1", "O2","P1", "P2", "Q1", "Q2"])
lr_full.fit(X, rtrain.results)

print('True Positives: ', end ='')
TP1=sum((lr_full.predict(X) == rtrain.results) & (lr_full.predict(X) == "g"))
print(TP1)
print('True Negatives: ', end ='')
TN1=sum((lr_full.predict(X) == rtrain.results) & (lr_full.predict(X) == "b"))
print(TN1)
print('False Positives: ', end ='')
FP1=sum((lr_full.predict(X) != rtrain.results) & (lr_full.predict(X) == "g"))
print(FP1)
print('False Negatives: ', end ='')
FN1=sum((lr_full.predict(X) != rtrain.results) & (lr_full.predict(X) == "b"))
print(FN1)
print('-'*100)

print('lr model on all variables except date accuracy on training data: ', end='')
print(sum(lr_full.predict(X) == rtrain.results) / len(rtrain))
print('sensitivity on training data:', end = '')
print(TP1/(TP1+FN1))
print('specificity on training data:', end = '')
print(TN1/(TN1+FP1))


rdev = pd.read_csv('dev-0/in.tsv', sep='\t', names=["A1", "A2", "B1", "B2", "C1", "C2","D1", "D2", "E1", "E2", "F1", "F2","G1", "G2", "H1", "H2", "I1", "I2","J1", "J2", "K1", "K2", "L1", "L2","M1", "M2", "N1", "N2", "O1", "O2","P1", "P2", "Q1", "Q2"])
rdev = pd.DataFrame(rdev, columns=["A1", "A2", "B1", "B2", "C1", "C2","D1", "D2", "E1", "E2", "F1", "F2","G1", "G2", "H1", "H2", "I1", "I2","J1", "J2", "K1", "K2", "L1", "L2","M1", "M2", "N1", "N2", "O1", "O2","P1", "P2", "Q1", "Q2"])
rdev2 = pd.read_csv('test-A/in.tsv', sep='\t', names=["A1", "A2", "B1", "B2", "C1", "C2","D1", "D2", "E1", "E2", "F1", "F2","G1", "G2", "H1", "H2", "I1", "I2","J1", "J2", "K1", "K2", "L1", "L2","M1", "M2", "N1", "N2", "O1", "O2","P1", "P2", "Q1", "Q2"])
rdev2 = pd.DataFrame(rdev2, columns=["A1", "A2", "B1", "B2", "C1", "C2","D1", "D2", "E1", "E2", "F1", "F2","G1", "G2", "H1", "H2", "I1", "I2","J1", "J2", "K1", "K2", "L1", "L2","M1", "M2", "N1", "N2", "O1", "O2","P1", "P2", "Q1", "Q2"])
rdev_expected = pd.read_csv('dev-0/expected.tsv', sep='\t', names=['y'])
print('-'*100)

print('results % :', end='')
print(sum(rdev_expected.y=="g") / len(rdev_expected))
print('zero rule model accuracy on dev set is',1 - sum(rdev_expected.y=="g") / len(rdev_expected))
print('-' * 100)

print('True Positives: ', end ='')
TP2=sum((lr_full.predict(rdev) == rdev_expected.y) & (lr_full.predict(rdev) == "g"))
print(TP2)
print('True Negatives: ', end ='')
TN2=sum((lr_full.predict(rdev) == rdev_expected['y']) & (lr_full.predict(rdev) == "b"))
print(TN2)
print('False Positives: ', end ='')
FP2=sum((lr_full.predict(rdev) != rdev_expected['y']) & (lr_full.predict(rdev) == "g"))
print(FP2)
print('False Negatives: ', end ='')
FN2=sum((lr_full.predict(rdev) != rdev_expected['y']) & (lr_full.predict(rdev) == "b"))
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

