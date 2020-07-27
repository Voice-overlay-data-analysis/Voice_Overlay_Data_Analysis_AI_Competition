from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
import numpy as np

X, Y = make_blobs(n_samples=200, centers=2, cluster_std=0.5, random_state=30) #데이터, #중심값
#200개의 무작위 데이터 클러스터 생성
#clusters : 클러스터 개수 (중심의 개수)
#cluster_std : 클러스터 표준편차
#Y값은 존재하지만 비지도 학습에서는 사용하지 않음

plt.scatter(X[:, 0], X[:, 1])

cluster_model = KMeans(n_clusters=3, init='random').fit(X)
#비지도 학습이라 Y값이 필요없음
#n_clusters : 군집화 할 클러스터링 개수
#init : 'random' 이면 KMeans, 'k-means++' 이면 KMeans++
center = cluster_model.cluster_centers_
# 점 하나 당 (x, y) 값
label = cluster_model.labels_
inertia = cluster_model.inertia_
#군집의 응집도, 값이 작을수록 군집도가 잘 되었다고 평가

c = ['r', 'g', 'b']
color = [c[i] for i in label]
plt.scatter(X[:, 0], X[:, 1], c=color)
for i in range(0, 3):
    plt.scatter(center[i][0], center[i][1], c=c[i], s=100, edgecolors='k')


op_k = -1
inertia = []
min_inertia = 10000
#최적의 k값 찾기
for i in range(2, 10):
    op_model = KMeans(n_clusters=i, init='random').fit(X)
    tmp = op_model.inertia_
    if tmp < min_inertia:
        inertia.append(tmp)
        op_k = i

inertia = np.array(inertia)
xaxis = np.array([i for i in range(2, 10)])
plt.plot(xaxis, inertia)
plt.scatter(xaxis, inertia)



