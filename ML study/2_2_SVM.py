"""===============================================
1. Linear SVM
==============================================="""
from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

data = make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0, n_classes=2, n_clusters_per_class=1, random_state=30)
X = data[0]
Y = data[1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

linear_SVM = SVC(kernel="linear", C=1).fit(X_train, Y_train)
#kernel="linear" : kernel을 안쓰는 것과 동일
support_vector_cnt = linear_SVM.n_support_
#각 클래스의 support vector 개수
support_vector = linear_SVM.support_vectors_
#각 클래스의 support vector의 x값 (x+, x-)

"""train data, support vector 시각화"""
plt.scatter(X_train[Y_train == 1, 0], X_train[Y_train == 1, 1], c='b', label='Class 1')
#Class 1이라고 예측한 데이터
plt.scatter(X_train[Y_train == 0, 0], X_train[Y_train == 0, 1], c='y', label='Class 0')
#Class 0이라고 예측한 데이터
plt.scatter(support_vector[:, 0], support_vector[:, 1], c='r', s=150, alpha=0.3)
#support vector
plt.xlabel("$X_1$")
plt.ylabel("$X_2$")
plt.title("Linear SVM")

"""hyperplane 시각화"""
#plt.contour 검색해보기
x1_min, x1_max = X_train[:, 0].min(), X_train[:, 0].max()
x2_min, x2_max = X_train[:, 1].min(), X_train[:, 1].max()
x1range = np.linspace(x1_min, x1_max)
x2range = np.linspace(x2_min, x2_max)
X1, X2 = np.meshgrid(x1range, x2range)
Z = np.empty(X1.shape)
for (i, j), val in np.ndenumerate(X1):
    x1 = val
    x2 = X2[i, j]
    p = linear_SVM.decision_function([[x1, x2]])
    Z[i, j] = p[0]
levels = [-1, 0, 1]
#결정경계(Z)는 Wx = 0, positive hyperplane은 Wx = +1, negative hyperplane은 Wx = -1
plt.contour(X1, X2, Z, levels, colors='k', linestyles=['dashed', 'solid', 'dashed'])


print("* train score : {}".format(linear_SVM.score(X_train, Y_train)))
print("* test score : {}".format(linear_SVM.score(X_test, Y_test)))

pred = linear_SVM.predict(X_test)
print("실제 Y_label : {}".format(Y_test))
print("예측 Y_label : {}".format(pred))
acc = np.sum([1 if pred[i] == Y_test[i] else 0 for i in range(len(pred))]) / len(pred)
print("* test score : {}".format(acc))

"""===============================================
2. Scaling - Standard Scaler
==============================================="""
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.fit_transform(X_test)

print("* scaling 전 RMS : {}".format(np.sum(np.square(X_train[:, 0])).mean()) + ", {}".format(np.sum(np.square(X_train[:, 1])).mean()))
print("* scaling 후 RMS : {}".format(np.sum(np.square(X_train_scale[:, 0])).mean()) + ", {}".format(np.sum(np.square(X_train_scale[:, 1])).mean()))
#RMS : Root Mean Square
#scaling 후 data의 범위가 달라짐

linear_SVM = SVC(kernel="linear", C=1).fit(X_train_scale, Y_train)
support_vector_cnt = linear_SVM.n_support_
support_vector = linear_SVM.support_vectors_

"""===============================================
3. Linear SVM Regularization
==============================================="""
reg_linear_SVM = SVC(kernel='linear', C=0.1).fit(X_train, Y_train)
support_vector_cnt = reg_linear_SVM.n_support_
support_vector = reg_linear_SVM.support_vectors_

print("* train scroe : {}".format(reg_linear_SVM.score(X_train, Y_train)))
print("* test score : {}".format(reg_linear_SVM.score(X_test, Y_test)))

