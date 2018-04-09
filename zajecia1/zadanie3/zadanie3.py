import pandas as pd
import seaborn as sns
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

r = pd.read_csv('/home/students/s441407/Desktop/umz-template/zajecia1/zadanie3/train/train.tsv', sep = '\t', names= ['price','isNew','rooms','floor','location','sqrMeters'])

r = r[r.price > 10.0]
r_cleared = r[r['price'] > 5000]

reg = linear_model.LinearRegression()

reg.fit(pd.DataFrame(r_cleared, columns=['sqrMeters', 'floor']), r_cleared['price'])

t = pd.read_csv('/home/students/s441407/Desktop/umz-template/zajecia1/zadanie3/dev-0/in.tsv', sep = '\t', names= ['isNew','rooms','floor','location','sqrMeters'])

Y_predict=reg.predict(pd.DataFrame(t, columns=['sqrMeters','floor']))

f = open('/home/students/s441407/Desktop/umz-template/zajecia1/zadanie3/dev-0/out.tsv', 'w')
for i in range(0, len(Y_predict)):
    f.write(str(Y_predict[i]))
    f.write('\n')
f.close()


