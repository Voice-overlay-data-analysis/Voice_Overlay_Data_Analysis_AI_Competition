from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
import mglearn

np.random.seed(30)
X = np.random.randn(100, 2)
Y = np.logical_xor(X[:, 0]>0, X[:, 1]>0)
Y = np.where(Y, 1, -1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

mlp_model = MLPClassifier(hidden_layer_sizes=([10, 5, 3]), max_iter=int(1e+05)).fit(X_train, Y_train)
#다층 퍼셉트론
#input layer와 output layer는 사용자가 정하는 것이 아님(데이터의 개수와 Class의 개수대로 정해짐)
print("* train scroe : {}".format(mlp_model.score(X_train, Y_train)))
print("* test score : {}".format(mlp_model.score(X_test, Y_test)))

weight = mlp_model.coefs_
bias = mlp_model.intercepts_
#len(weight) = len(bias) = 입력층 + hidden layer 수

for i, w in enumerate(weight):
    print("{}번째 hidden layer의 weight 개수 : {}".format(i, w.shape))
#0번째 hidden layer = input layer

mglearn.plots.plot_2d_separator(mlp_model, X_train, fill=True, alpha=0.1)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], Y_train)

mglearn.plots.plot_2d_separator(mlp_model, X_test, fill=True, alpha=0.1)
mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], Y_test)

