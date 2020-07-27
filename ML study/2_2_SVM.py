"""===============================================
1. Linear SVM
==============================================="""
from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from module import *

data = make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0, n_classes=2, n_clusters_per_class=1, random_state=30)
X = data[0]
Y = data[1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

linear_SVM = SVC(kernel="linear", C=1).fit(X_train, Y_train)
#kernel="linear" : kernel을 안쓰는 것과 동일
linear_support_vector_cnt = linear_SVM.n_support_
#각 클래스의 support vector 개수
linear_support_vector = linear_SVM.support_vectors_
#각 클래스의 support vector의 x값 (x+, x-)

print("* train score : {}".format(linear_SVM.score(X_train, Y_train)))
print("* test score : {}".format(linear_SVM.score(X_test, Y_test)))

pred = linear_SVM.predict(X_test)
print("실제 Y_label : {}".format(Y_test))
print("예측 Y_label : {}".format(pred))
acc = np.sum([1 if pred[i] == Y_test[i] else 0 for i in range(len(pred))]) / len(pred)
print("* test score : {}".format(acc))

"""===============================================
2. Linear SVM Regularization
==============================================="""
reg_linear_SVM = SVC(kernel='linear', C=0.1).fit(X_train, Y_train)
support_vector_cnt = reg_linear_SVM.n_support_
support_vector = reg_linear_SVM.support_vectors_

print("* train scroe : {}".format(reg_linear_SVM.score(X_train, Y_train)))
print("* test score : {}".format(reg_linear_SVM.score(X_test, Y_test)))

"""===============================================
3. Non-Linear SVM (Kernel SVM)
==============================================="""
kernel_SVM = SVC(kernel='poly', C=1, degree=1, coef0=1, gamma=1).fit(X_train, Y_train)
#kernel='kernel : 다항식 커널 (비선형)
#ploy로 설정 시, degree 다항식 차수 지정 (default : 3), 모델복잡도를 결정
#coef0 =높은 차수와 낮은 차수의 영향력, coef0이 커지면 고차항의 영향을 많이 받게됨
#gamma = C

kernel_support_vector_cnt = kernel_SVM.n_support_
kernel_support_vector = kernel_SVM.support_vectors_

print("* train score : {}".format(kernel_SVM.score(X_train, Y_train)))
print("* test score : {}".format(kernel_SVM.score(X_test, Y_test)))

"""===============================================
4. 시각화
==============================================="""
#draw_contour는 module.py에 정의되어 있음
draw_contour(linear_SVM, linear_support_vector, X_train, Y_train, 'train')
draw_contour(linear_SVM, linear_support_vector, X_test, Y_test, 'test')

draw_contour(kernel_SVM, kernel_support_vector, X_train, Y_train, 'train')
draw_contour(kernel_SVM, kernel_support_vector, X_test, Y_test, 'test')

"""===============================================
4. 성능 비교
==============================================="""
print("<Linear SVM>")
print("* train score : {}".format(linear_SVM.score(X_train, Y_train)))
print("* test score : {}".format(linear_SVM.score(X_test, Y_test)))

print("\n<Linear Regulated SVM>")
print("* train score : {}".format(reg_linear_SVM.score(X_train, Y_train)))
print("* test score : {}".format(reg_linear_SVM.score(X_test, Y_test)))

print("\n<Kernel SVM>")
print("* train score : {}".format(kernel_SVM.score(X_train, Y_train)))
print("* test score : {}".format(kernel_SVM.score(X_test, Y_test)))

"""===============================================
4. Scaling - Standard Scaler
==============================================="""
from sklearn.preprocessing import StandardScaler
#평균 0, 분산 1로 정규화

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

