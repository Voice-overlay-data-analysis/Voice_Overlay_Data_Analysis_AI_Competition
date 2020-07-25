"""feature 개수가 많아지면 Linear Regression의 Overfitting 확률이 올라가므로
모델 학습에 제한을 거는 모델들"""

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
import numpy as np

data = make_regression(n_samples=100, n_features=70, bias=10, noise=100, random_state=100)
X = data[0]
Y = data[1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

"""===============================================
1. linear, ridge, lasso 모델 비교
==============================================="""
linear_model = LinearRegression().fit(X_train, Y_train)
print("<Linear>\n* train_score : {}\n".format(linear_model.score(X_train, Y_train)) + "* test_score : {}\n".format(linear_model.score(X_test, Y_test)) + "* weight 크기 : {}".format(np.square(linear_model.coef_).mean()))
ridge_model = Ridge().fit(X_train, Y_train)
print("<Ridge>\n* train_score : {}\n".format(ridge_model.score(X_train, Y_train)) + "* test_score : {}\n".format(ridge_model.score(X_test, Y_test)) + "* weight 크기 : {}".format(np.square(ridge_model.coef_).mean()))
lasso_model = Lasso().fit(X_train, Y_train)
print("<Lasso>\n* train_score : {}\n".format(lasso_model.score(X_train, Y_train)) + "* test_score : {}\n".format(lasso_model.score(X_test, Y_test)) + "* weight 크기 : {}".format(np.square(lasso_model.coef_).mean()))

#weight 크기(모델 복잡도) 확인해보기!
#weight 크기가 크면 Overfitting 일어날 확률이 높아진다
#train 성능은 좋은데 test 성능이 안좋다면 Overfit
#train, test 둘 다 성능이 안좋으면 Underfit

lasso_weights = lasso_model.coef_
#Lasso의 weight들 중에서 0이 되어버린 값이 존재함
lasso_feature = np.sum(lasso_model.coef_ != 0)
#Lasso Regression에서 실제 모델링에 사용된 특징 개수

"""===============================================
2. ridge alpha 값 조절해보기
==============================================="""
#default alpha = 1.0
alpha = 1
ridge_model = Ridge(alpha).fit(X_train, Y_train)
print("<Ridge({})>".format(alpha))
print("* train_score : {}\n".format(ridge_model.score(X_train, Y_train)) + "* test_score : {}\n".format(ridge_model.score(X_test, Y_test)) + "* weight 크기 : {}".format(np.square(ridge_model.coef_).mean()))

#alpha = 10
alpha = 10
ridge_high_model = Ridge(alpha=alpha).fit(X_train, Y_train)
print("\n<Ridge({})>".format(alpha))
print("* train_score : {}\n".format(ridge_high_model.score(X_train, Y_train)) + "* test_score : {}\n".format(ridge_high_model.score(X_test, Y_test)) + "* weight 크기 : {}".format(np.square(ridge_high_model.coef_).mean()))

#alpha = 0.1
alpha = 0.1
ridge_low_model = Ridge(alpha=alpha).fit(X_train, Y_train)
print("\n<Ridge({})>".format(alpha))
print("* train_score : {}\n".format(ridge_low_model.score(X_train, Y_train)) + "* test_score : {}\n".format(ridge_low_model.score(X_test, Y_test)) + "* weight 크기 : {}".format(np.square(ridge_low_model.coef_).mean()))

#alpha값을 높이면 모델 복잡도에 예민해지므로 훈련 셋의 성능은 나빠지지만 일반화에 도움을 줌
#alpha값을 낮추면 제한하는 정도가 낮아지므로 선형회귀와 비슷해짐

#Ridge Regression을 이용하여 어느 정도 train 성능(>0.8)을 보장하면서 높일 수 있는 test 성능의 최고치 구하기
optimize_alpha = 0
diff = 100
optimize_score = [0, 0]
for i in range(1, 100):
    tmp = Ridge(alpha=i).fit(X_train, Y_train)
    train_sc = tmp.score(X_train, Y_train)
    test_sc = tmp.score(X_test, Y_test)
    if train_sc > 0.8:
        if train_sc - test_sc < diff:
            optimize_alpha = i
            diff = tmp.score(X_train, Y_train) - tmp.score(X_test, Y_test)
            optimize_score[0], optimize_score[1] = train_sc, test_sc

alpha = optimize_alpha
print("\n<Ridge({})>".format(alpha))
print("* train_score : {}\n".format(optimize_score[0]) + "* test_score : {}\n".format(optimize_score[1]))

"""===============================================
3. lasso alpha 값 조절해보기
==============================================="""
#default alpha = 1.0
alpha = 1
lasso_model = Lasso(alpha=alpha).fit(X_train, Y_train)
print("<Lasso({})>".format(alpha))
print("* train_score : {}\n".format(lasso_model.score(X_train, Y_train)) + "* test_score : {}\n".format(lasso_model.score(X_test, Y_test)) + "* weight 크기 : {}".format(np.square(ridge_model.coef_).mean()))

#alpha = 10
alpha = 10
ridge_high_model = Lasso(alpha=alpha).fit(X_train, Y_train)
print("\n<Lasso({})>".format(alpha))
print("* train_score : {}\n".format(ridge_high_model.score(X_train, Y_train)) + "* test_score : {}\n".format(ridge_high_model.score(X_test, Y_test)) + "* weight 크기 : {}".format(np.square(ridge_high_model.coef_).mean()))

#alpha = 0.1
alpha = 0.1
ridge_low_model = Lasso(alpha=alpha).fit(X_train, Y_train)
print("\n<Lasso({})>".format(alpha))
print("* train_score : {}\n".format(ridge_low_model.score(X_train, Y_train)) + "* test_score : {}\n".format(ridge_low_model.score(X_test, Y_test)) + "* weight 크기 : {}".format(np.square(ridge_low_model.coef_).mean()))

#Lasso Regression을 이용하여 어느 정도 train 성능(>0.8)을 보장하면서 높일 수 있는 test 성능의 최고치 구하기
optimize_alpha = 0
diff = 100
optimize_score = [0, 0]
feature_cnt = 0
for i in range(1, 100):
    tmp = Lasso(alpha=i).fit(X_train, Y_train)
    train_sc = tmp.score(X_train, Y_train)
    test_sc = tmp.score(X_test, Y_test)
    if train_sc > 0.8:
        if train_sc - test_sc < diff:
            optimize_alpha = i
            diff = tmp.score(X_train, Y_train) - tmp.score(X_test, Y_test)
            optimize_score[0], optimize_score[1] = train_sc, test_sc
            feature_cnt = np.sum(tmp.coef_ != 0)

alpha = optimize_alpha
print("\n<Lasso({})>".format(alpha))
print("* train_score : {}\n".format(optimize_score[0]) + "* test_score : {}\n".format(optimize_score[1]))
print("* feature cnt : {}".format(feature_cnt))

"""===============================================
4. Elasitc Net
==============================================="""
elastic_model = ElasticNet().fit(X_train, Y_train)
print("<Elastic>")
print("* train_score : {}\n".format(elastic_model.score(X_train, Y_train)) + "* test_score : {}".format(elastic_model.score(X_test, Y_test)))
print("* feature cnt : {}".format(np.sum(elastic_model.coef_ != 0)))

#alpha = L1_ratio + L2_ratio (L1_norm 과 L2_norm의 비율)
#L1_norm : Lasso weight, L2_norm : Ridge weight
elastic_high_model = ElasticNet(alpha=10, l1_ratio=0.7).fit(X_train, Y_train)


#======================================================================
#======================================================================
#보스턴 주택데이터 예측하기
#======================================================================
#======================================================================
from mglearn.datasets import load_extended_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

boston = load_extended_boston()
X = boston[0]
Y = boston[1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

alpha = 0.001
linear_model = LinearRegression().fit(X_train, Y_train)
print("<Linear>\n* train_score : {}\n".format(linear_model.score(X_train, Y_train)) + "* test_score : {}\n".format(linear_model.score(X_test, Y_test)) + "* weight 크기 : {}".format(np.square(linear_model.coef_).mean()))
ridge_model = Ridge(alpha=alpha).fit(X_train, Y_train)
print("\n<Ridge>\n* train_score : {}\n".format(ridge_model.score(X_train, Y_train)) + "* test_score : {}\n".format(ridge_model.score(X_test, Y_test)) + "* weight 크기 : {}".format(np.square(ridge_model.coef_).mean()))
alpha = 0.1
lasso_model = Lasso(alpha=alpha).fit(X_train, Y_train)
print("\n<Lasso>\n* train_score : {}\n".format(lasso_model.score(X_train, Y_train)) + "* test_score : {}\n".format(lasso_model.score(X_test, Y_test)) + "* weight 크기 : {}\n".format(np.square(lasso_model.coef_).mean()) + "* feature cnt : {}".format(np.sum(lasso_model.coef_ != 0)))
elastic_model = ElasticNet(alpha=alpha).fit(X_train, Y_train)
print("\n<Elastic>\n* train_score : {}\n".format(elastic_model.score(X_train, Y_train)) + "* test_score : {}\n".format(elastic_model.score(X_test, Y_test)) + "* weight 크기 : {}\n".format(np.square(elastic_model.coef_).mean()) + "* feature cnt : {}".format(np.sum(elastic_model.coef_ != 0)))





