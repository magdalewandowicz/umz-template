import pandas as pd
import graphviz
from sklearn import tree
from sklearn.metrics import confusion_matrix

train = pd.read_csv('train.csv', sep=',')
test = pd.read_csv('test.csv', sep=',')

train_X = pd.DataFrame(train, columns=train.columns[:-1])
train_Y = train['Class']

test_X = pd.DataFrame(test, columns=test.columns[:-1])
test_Y = test['Class']


def show_plot():
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=train.columns[:-1],
                                    class_names=[str(x)
                                                 for x in [1, 2, 3, 4, 5, 6, 7]],
                                    filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    graph.view()


clf = tree.DecisionTreeClassifier(max_depth=5,min_samples_split=10)
clf = clf.fit(train_X, train_Y)

#clf = tree.DecisionTreeClassifier(criterion="entropy")
#clf = clf.fit(train_X, train_Y)

#clf = tree.DecisionTreeClassifier(min_weight_fraction_leaf=0.1)
#clf = clf.fit(train_X, train_Y)

show_plot()

print('TRAIN SET')
print(confusion_matrix(clf.predict(train_X), train_Y))
print(sum(clf.predict(train_X) == train_Y) / len(train_X))
print('TEST SET')
print(confusion_matrix(clf.predict(test_X), test_Y))
print(sum(clf.predict(test_X) == test_Y) / len(test_X))
