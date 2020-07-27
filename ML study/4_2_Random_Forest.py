from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from module import *

iris = load_iris()
X = iris['data']
Y = iris['target']
X_train, X_test, Y_train, Y_test = train_test_split(X[:, : 2], Y, test_size=0.2, shuffle=True)

dt_model = DecisionTreeClassifier(random_state = 0).fit(X_train, Y_train)
random_forest_model1 = BaggingClassifier(DecisionTreeClassifier(random_state=10), n_estimators=100, max_samples=40).fit(X_train, Y_train)
#Random Forest는 Decision Tree를 Bagging 방법으로 합친 앙상블 모델
#n_estimators : 몇 개의 모델을 앙상블해서 훈련시킬 지 결정 (default = 10)
#max_samples : 샘플 데이터 수
random_forest_model2 = RandomForestClassifier(n_estimators=100, random_state=10, max_samples=40).fit(X_train, Y_train)

print("<Decision Tree>")
print("* train score : {}".format(dt_model.score(X_train, Y_train)))
print("* test score : {}".format(dt_model.score(X_test, Y_test)))

print("\n<Random Forest1>")
print("* train score : {}".format(random_forest_model1.score(X_train, Y_train)))
print("* test score : {}".format(random_forest_model1.score(X_test, Y_test)))

print("\n<Random Forest2>")
print("* train score : {}".format(random_forest_model2.score(X_train, Y_train)))
print("* test score : {}".format(random_forest_model2.score(X_test, Y_test)))

model_border(dt_model, X_train, Y_train)
model_border(random_forest_model1, X_train, Y_train)
model_border(random_forest_model2, X_train, Y_train)

#약한 학습기 시각화
plt.figure(figsize=(30, 20))
for i in range(0, 10):
    plt.subplot(3, 4, i+1)
    plt.title("sub tree {}".format(i+1))
    model_border(random_forest_model2.estimators_[i], X_train, Y_train)
plt.subplot(3, 4, 11)
plt.title("Random Forest tree")
model_border(random_forest_model2, X_train, Y_train)