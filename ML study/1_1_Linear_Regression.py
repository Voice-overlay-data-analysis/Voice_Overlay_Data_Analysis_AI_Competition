"""===============================================
1-1. data load
==============================================="""
from sklearn.datasets import make_regression
data = make_regression(n_samples=1000, n_features=1, bias=1000, noise=10)
#make_regression() : 회귀분석용 데이터를 생성하는 명령어
#1000개의 데이터, 1개의 특징을 가진 데이터셋 생성
# noise: 해당 가설로 예측할 수 없는 모든 영향에 대한 오차

X = data[0]
Y = data[1]
"""===============================================
1-2. data split
==============================================="""
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
"""===============================================
2. Declare Model
==============================================="""
from sklearn.linear_model import LinearRegression
model = LinearRegression()
"""===============================================
3. Model train
==============================================="""
model = model.fit(X_train, Y_train)
# 1) weight, bias 초기값 설정
# 2) predict = w`x + b` 구하고 cost 계산
# 3) w에 대한 cost의 기울기를 구하여 경사하강법을 통해 cost를 가장 낮게 하는 최적의 w 구하기
"""===============================================
4. Check score
==============================================="""
train_score = model.score(X_train, Y_train)
test_score = model.score(X_test, Y_test)
#score() 실행 시 내부적으로 X_test를 이용하여 예측값(predict_data) 생성
#그 후 Linear Regression R2_Score를 통하여 모델 성능 계산
#train_score, test_score를 비교하여 overfit 여부를 알 수 있음
"""===============================================
5. Check result(weight)
==============================================="""
weight = model.coef_
bias = model.intercept_
"""===============================================
6. Visualize Model
==============================================="""
from matplotlib import pyplot as plt
res_hypothesis = weight*X + bias
plt.scatter(X, Y)
plt.plot(X, res_hypothesis, c="r")

#======================================================================
#======================================================================
#보스턴 주택데이터 예측하기
#======================================================================
#======================================================================
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
boston = load_boston()
keys = boston.keys()
# >> dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])
"""
data = x
target = y
feature_names = x 카테고리명
DESCR = 데이터에 대한 설명
filename = 파일 경로
"""
X = boston['data']
Y = boston['target']
X_features = boston['feature_names']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
model = LinearRegression()
model = model.fit(X_train, Y_train)
model.score(X_train, Y_train), model.score(X_test, Y_test)
weight = model.coef_
bias = model.intercept_
#len(weight) = len(X_features)