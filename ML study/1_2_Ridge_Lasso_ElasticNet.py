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
print("<Linear>")
print("* train_score : {}\n".format(linear_model.score(X_train, Y_train)))
print("* test_score : {}\n".format(linear_model.score(X_test, Y_test)))
print("* weight 크기 : {}".format(np.square(linear_model.coef_).mean()))
ridge_model = Ridge().fit(X_train, Y_train)
print("<Ridge>")
print("* train_score : {}\n".format(ridge_model.score(X_train, Y_train)))
print("* test_score : {}\n".format(ridge_model.score(X_test, Y_test)))
print("* weight 크기 : {}".format(np.square(ridge_model.coef_).mean()))
lasso_model = Lasso().fit(X_train, Y_train)
print("<Lasso>")
print("* train_score : {}\n".format(lasso_model.score(X_train, Y_train)))
print("* test_score : {}\n".format(lasso_model.score(X_test, Y_test)))
print("* weight 크기 : {}".format(np.square(lasso_model.coef_).mean()))
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
#alpha값을 높이면 모델 복잡도에 예민해지므로 훈련 셋의 성능은 나빠지지만 일반화에 도움을 줌
#alpha값을 낮추면 제한하는 정도가 낮아지므로 선형회귀와 비슷해짐
for i in [1, 10, 0.1]:
    compare_model = Ridge(alpha=i).fit(X_train, Y_train)
    print("<Ridge (alpha = {})>".format(i))
    print("* train_score : {}".format(compare_model.score(X_train, Y_train)))
    print("* test_score : {}\n".format(compare_model.score(X_test, Y_test)))

"""===============================================
3. lasso alpha 값 조절해보기
==============================================="""
for i in [1, 10, 0.1]:
    compare_model = Lasso(alpha=i).fit(X_train, Y_train)
    print("<Ridge (alpha = {})>".format(i))
    print("* train_score : {}".format(compare_model.score(X_train, Y_train)))
    print("* test_score : {}\n".format(compare_model.score(X_test, Y_test)))

"""===============================================
4. Elasitc Net
==============================================="""
elastic_model = ElasticNet().fit(X_train, Y_train)
print("<Elastic>")
print("* train_score : {}".format(elastic_model.score(X_train, Y_train)))
print("* test_score : {}\n".format(elastic_model.score(X_test, Y_test)))
print("* feature cnt : {}".format(np.sum(elastic_model.coef_ != 0)))

#alpha = L1_ratio + L2_ratio (L1_norm 과 L2_norm의 비율)
#L1_norm : Lasso weight, L2_norm : Ridge weight
elastic10_model = ElasticNet(alpha=10, l1_ratio=0.7).fit(X_train, Y_train)