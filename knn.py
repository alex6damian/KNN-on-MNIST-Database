import numpy as np
import operator
from operator import itemgetter
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

def euclidian_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, K=3):
        self.K = K
    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.Y_train = y_train

    def predict(self, X_test):
        predictions = []
        for i in range(len(X_test)):
            dist = np.array([euclidian_distance(X_test[i], x_t) for x_t in
                             self.X_train])
            dist_sorted = dist.argsort()[:self.K]
            neigh_count = {}
            for idx in dist_sorted:
                if self.Y_train[idx] in neigh_count:
                    neigh_count[self.Y_train[idx]] += 1
                else:
                    neigh_count[self.Y_train[idx]] = 1
            sorted_neigh_count = sorted(neigh_count.items(),
                                        key=operator.itemgetter(1), reverse=True)
            predictions.append(sorted_neigh_count[0][0])
        return predictions


mnist=load_digits()
print(mnist.data.shape)

X=mnist.data
y=mnist.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

kVals = np.arange(3,100,2)
accuracies=[]
for kval in kVals:
    model=KNN(K=kval)
    model.fit(X_train, y_train)
    pred=model.predict(X_test)
    acc=accuracy_score(y_test, pred)
    accuracies.append(acc)
    print("K=%d, accuracy=%.2f%%" % (kval, acc*100))

max_index=accuracies.index(max(accuracies))
print(max_index)

from matplotlib import pyplot as plt
plt.plot(kVals, accuracies)
plt.xlabel('K Value')
plt.ylabel('Accuracy')

model=KNN(K=3)
model.fit(X_train, y_train)
pred=model.predict(X_test)
acc=accuracy_score(y_test, pred)

precision, recall, fscore, _ = precision_recall_fscore_support(y_test, pred)

print('Precision \n', precision)
print('\nRecall \n', recall)
print('\nF-score \n', fscore)

print(acc)