from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import mglearn # 딥러닝 오픈소스
import numpy as np

np.random.seed(30)
X = np.random.randn(100, 2)
#feature 수 : 2, 전체 데이터 개수 : 100*2
Y = np.logical_xor(X[:, 0]>0, X[:, 1]>0)
# X[i][0] > 0 이면 True, 아니면 False
# X[i][1] > 0 이면 True, 아니면 False
# Y에 이 두개의 값을 XOR 연산한 결과값 저장
Y = np.where(Y, 1, -1)
#True이면 1, False이면 -1로 치환
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

perceptron_model = MLPClassifier(hidden_layer_sizes=()).fit(X_train, Y_train)
#단일 퍼셉트론
print("* train score : {}".format(perceptron_model.score(X_train, Y_train)))
print("* test score : {}".format(perceptron_model.score(X_test, Y_test)))
#단일 퍼셉트론은 xor(비선형문제)을 설명할 수없음
#and, or(선형문제)는 해결 가능

weight = perceptron_model.coefs_
#각 층의 weight 개수
bias = perceptron_model.intercepts_

mglearn.plots.plot_2d_separator(perceptron_model, X_train, fill=True, alpha=0.1)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], Y_train)

mglearn.plots.plot_2d_separator(perceptron_model, X_test, fill=True, alpha=0.1)
mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], Y_test)
