from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt

"""===============================================
1. Logistic Regression
==============================================="""
data = make_classification(n_samples=100, n_features=1, n_informative=1, n_redundant=0, n_clusters_per_class=1, random_state=30)
#make_classification() : 분류데이터를 생성하는 명령어
#n_informative : 독립변수 중 종속변수와 상관관계가 있는 성분의 수 (n_features >= n_informative)
#n_redundant : 독립변수의 선형조합으로 만들어지는 독립변수의 수 (default=2)
#n_clusters_per_class : 각 클래스당 클러스터 수, (default=2, class수 * n_cluster_per_class <= 2^n_informative)
#random_state : 난수

X = data[0]
Y = data[1]

#train data 시각화
plt.figure(figsize=(10, 5))
plt.scatter(X, Y, marker="*", c=Y, edgecolors="k", s=200)
#marker : 모양, c : 색깔, edgecolors : 테두리, s : 크기
plt.xlabel('data')
plt.ylabel('label')
plt.title('Logistic Regression (Feature cnt : 1)')

#데이터 분리
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

#성능 평가 방법 1.
logistic_model = LogisticRegression().fit(X_train, Y_train)
weight = logistic_model.coef_
bias = logistic_model.intercept_
print("<Linear>\n* train_score : {}\n".format(logistic_model.score(X_train, Y_train)) + "* test_score : {}\n".format(logistic_model.score(X_test, Y_test)) + "* weight 크기 : {}".format(np.square(logistic_model.coef_).mean()))

#성능 평가 방법 2.
pred = logistic_model.predict(X_test)
acc = np.sum([1 if pred[i] == Y_test[i] else 0 for i in range(len(Y_test))]) / len(Y_test)
print("* test_score : {}".format(acc))

#시각화 - logistic 함수 그리기
xaxis = np.linspace(-3, 3, 100)
#x축 범위 설정
res_hypothesis = weight[0][0]*xaxis + bias[0]
#모델링을 통해 구한 hypothesis
logistic_func = 1/(1 + np.exp(-res_hypothesis))
#구한 hypothesis를 (0, 1) 범위로 변환하며 logistic function 구하기
plt.plot(xaxis, logistic_func)

#시각화 - 예측 데이터 시각화
plt.scatter(X_test, Y_test, c=pred, edgecolors='k', s=300)
plt.colorbar(label='predict result')
plt.xlabel('Feature (X)')
plt.ylabel('lable (Y)')

#label값이 0인데 노란점 -> 실제 label 0인데 label 1로 잘못 예측
#label값이 1인데 보라점 -> 실제 label 1인데 label 0으로 잘못 예측

"""===============================================
2. Logistic Regression의 Regularization(규제화)
==============================================="""
data = make_classification(n_samples=100, n_features=10, n_informative=1, n_redundant=3, n_clusters_per_class=1, random_state=30, n_repeated=0)
X = data[0]
Y = data[1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

reg_model = LogisticRegression(penalty='l1', C=1.5, solver='saga').fit(X_train, Y_train)
#penalty : l1(Lasso), l2(Ridge), (default : l2)
#C : 규제강도, 오차와 모델복잡도의 비율 (오차에 곱해짐)
#C값이 높아지면 모델 복잡도에 덜 신경씀(규제 감소), 데이터를 정확하게 분류하지만 Overfitting이 일어날 수 있음

"""===============================================
2. 유방암 데이터 실습
==============================================="""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

breast = load_breast_cancer()
keys = breast.keys()
X = breast['data']
Y = breast['target']
kind_of_label = breast['target_names']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

logistic_model = LogisticRegression(C=5).fit(X_train, Y_train)
print("* train score : {}".format(logistic_model.score(X_train, Y_train)))
print("* test score : {}".format(logistic_model.score(X_test, Y_test)))

xaxis = np.linspace(1, 10, 10)
train_score = []
test_score = []
for i in range(1, 11):
    logistic_model = LogisticRegression(C=i).fit(X_train, Y_train)
    train_score.append(logistic_model.score(X_train, Y_train))
    test_score.append(logistic_model.score(X_test, Y_test))
plt.plot (xaxis, train_score, c='b', marker='.')
plt.plot(xaxis, test_score, c='r', marker='.')
plt.xticks(np.arange(1, 11, 1))
plt.ylim(0.9, 1.0)
plt.xlabel("C value")
plt.ylabel("Score")
plt.title("Logistic Regression Regulation")


"""===============================================
3. SoftMax Regression(Multinomial Logistic Regrssion
==============================================="""


