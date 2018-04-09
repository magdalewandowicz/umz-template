import pandas as pd
import seaborn as sns
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

r = pd.read_csv('/home/students/s441407/Desktop/umz-template/zajecia1/zadanie4/train/in.tsv', sep = '\t', names= ['price','mileage', 'year','brand','engineType','engineCapacity'])

r = r[r.price > 10.0]
r_cleared = r[r['price'] > 500]

reg = linear_model.LinearRegression()


reg.fit(pd.DataFrame(r_cleared, columns=['year', 'mileage', 'engineCapacity']), r_cleared['price'])

t = pd.read_csv('/home/students/s441407/Desktop/umz-template/zajecia1/zadanie4/dev-0/in.tsv', sep = '\t', names= ['mileage', 'year','brand','engineType','engineCapacity'])
Y_predict=reg.predict(pd.DataFrame(t, columns=['year', 'mileage', 'engineCapacity']))

f = open('/home/students/s441407/Desktop/umz-template/zajecia1/zadanie4/dev-0/out.tsv', 'w')
for i in range(0, len(Y_predict)):
    f.write(str(Y_predict[i]))
    f.write('\n')
f.close()


