# Prediction

import pandas as pd
import numpy as np
import seaborn as sb
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

#============================================================
#data load
#============================================================

#1,2,3,4 열만 load
advertising = pd.read_csv('./Advertising.csv', usecols=[1, 2, 3,4])
#load한 데이터 확인(첫 10줄만)
advertising.head()
#load한 데이터 확인(마지막 10줄만)
advertising.tail()
#load한 데이터 구조 확인
advertising.shape
#load한 데이터 카테고리 출력
advertising.columns
#load한 데이터를 그래프로 시각화
sb.pairplot(advertising)

#============================================================
#TV예측 변수를 이용하여 판매량 반응 변수 예측 - ols
#============================================================

lm = smf.ols(formula='sales ~ TV', data=advertising)
#<statsmodels.regression.linear_model.OLS object at 0x000001DBBB2CD400>
lm_hat  = lm.fit()

#============================================================
#시각화
#============================================================

#제목
plt.title("Simple Linear Regression")
#x축
plt.xlabel("TV")
#y축
plt.ylabel("Sales")
#산점도(scatter plot) 그리기
plt.scatter(advertising.TV, advertising.sales)
plt.show()

#모델링한 직선 시각화
X = pd.DataFrame({'TV':[advertising.TV.min(), advertising.TV.max()]})
Y_pred = lm_hat.predict(X)
plt.plot(X, Y_pred, c='r')
