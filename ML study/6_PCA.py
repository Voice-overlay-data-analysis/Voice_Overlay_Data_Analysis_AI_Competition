from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import mglearn

cancer = load_breast_cancer()
X = cancer['data']
Y = cancer['target']
#x_feature 개수가 너무 많으니 차원 축소가 필요함

#차원 축소 하기 전에 scaling
X = StandardScaler().fit(X).transform(X)

#차원 축소 하기 위해 주성분 분석(PCA)
pca_X = PCA(n_components=2).fit(X).transform(X)
#n_compoments : 주성분 개수
#x_feature가 30개에서 2개로 차원 축소

#차원 축소 결과로 시각화
plt.scatter(pca_X[:, 0], pca_X[:, 1], c=Y)
