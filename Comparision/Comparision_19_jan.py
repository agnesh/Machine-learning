#Comparision
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log2

df = pd.read_csv("C:/Users/agnes/OneDrive/Desktop/ML/fertility.csv")

for col in df.columns:
    if df[col].dtype == object:
        df[col] = np.unique(df[col], return_inverse=True)[1]

X = df.iloc[:, :-1].values.astype(float)
y = df.iloc[:, -1].values.astype(int)

X = (X - X.mean(axis=0)) / X.std(axis=0)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

def acc(a, b):
    return np.mean(a == b)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

scores = {}

X_lr = np.c_[np.ones(len(X_train)), X_train]
X_lr_t = np.c_[np.ones(len(X_test)), X_test]
w = np.zeros(X_lr.shape[1])

for _ in range(1000):
    w -= 0.01 * (X_lr.T @ (sigmoid(X_lr @ w) - y_train)) / len(y_train)

lr_pred = sigmoid(X_lr_t @ w) >= 0.5
scores["Logistic Regression"] = acc(y_test, lr_pred)
print("Logistic Regression Accuracy:", scores["Logistic Regression"])

def knn(Xtr, ytr, Xte, k=5):
    r = []
    for x in Xte:
        d = np.sqrt(((Xtr - x) ** 2).sum(axis=1))
        idx = np.argsort(d)[:k]
        v, c = np.unique(ytr[idx], return_counts=True)
        r.append(v[c.argmax()])
    return np.array(r)

scores["KNN"] = acc(y_test, knn(X_train, y_train, X_test))
print("KNN Accuracy:", scores["KNN"])

class GNB:
    def fit(self, X, y):
        self.c = np.unique(y)
        self.m = {}
        self.v = {}
        self.p = {}
        for i in self.c:
            Xc = X[y == i]
            self.m[i] = Xc.mean(axis=0)
            self.v[i] = Xc.var(axis=0) + 1e-9
            self.p[i] = len(Xc) / len(X)

    def predict(self, X):
        out = []
        for x in X:
            post = []
            for i in self.c:
                a = -0.5 * np.sum(np.log(2 * np.pi * self.v[i]))
                b = -0.5 * np.sum(((x - self.m[i]) ** 2) / self.v[i])
                post.append(np.log(self.p[i]) + a + b)
            out.append(self.c[np.argmax(post)])
        return np.array(out)

nb = GNB()
nb.fit(X_train, y_train)
scores["Naive Bayes"] = acc(y_test, nb.predict(X_test))
print("Naive Bayes Accuracy:", scores["Naive Bayes"])

def entropy(y):
    p = np.bincount(y) / len(y)
    return -np.sum([i * log2(i) for i in p if i > 0])

def split_data(X, y, f, t):
    m = X[:, f] <= t
    return X[m], y[m], X[~m], y[~m]

def best_split(X, y):
    bg, bi, bt = -1, None, None
    for i in range(X.shape[1]):
        for t in np.unique(X[:, i]):
            _, yl, _, yr = split_data(X, y, i, t)
            if len(yl) == 0 or len(yr) == 0:
                continue
            g = entropy(y) - (len(yl)/len(y))*entropy(yl) - (len(yr)/len(y))*entropy(yr)
            if g > bg:
                bg, bi, bt = g, i, t
    return bi, bt

def build_tree(X, y, d=0):
    if len(np.unique(y)) == 1 or d == 5:
        return np.bincount(y).argmax()
    i, t = best_split(X, y)
    if i is None:
        return np.bincount(y).argmax()
    Xl, yl, Xr, yr = split_data(X, y, i, t)
    return (i, t, build_tree(Xl, yl, d+1), build_tree(Xr, yr, d+1))

def predict_tree(x, tr):
    if not isinstance(tr, tuple):
        return tr
    i, t, l, r = tr
    return predict_tree(x, l) if x[i] <= t else predict_tree(x, r)

tree = build_tree(X_train, y_train)
dt_pred = np.array([predict_tree(x, tree) for x in X_test])
scores["Decision Tree"] = acc(y_test, dt_pred)
print("Decision Tree Accuracy:", scores["Decision Tree"])

plt.figure()
plt.bar(scores.keys(), scores.values())
plt.title("Classification Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.show()

Xr = np.c_[np.ones(len(X_train)), X_train]
Xr_t = np.c_[np.ones(len(X_test)), X_test]
w_reg = np.linalg.inv(Xr.T @ Xr) @ Xr.T @ y_train
y_reg = Xr_t @ w_reg

mse = np.mean((y_test - y_reg) ** 2)
print("Multiple Linear Regression MSE:", mse)

plt.figure()
plt.plot(y_test, label="Actual")
plt.plot(y_reg, label="Predicted")
plt.title("Multiple Linear Regression")
plt.legend()
plt.show()
