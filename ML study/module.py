import numpy as np
from matplotlib import pyplot as plt

def draw_contour(model, support_vector, X, Y, tag):
    """train data, support vector 시각화"""
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], c='b', label='Class 1')
    # Class 1이라고 예측한 데이터
    plt.scatter(X[Y == 0, 0], X[Y == 0, 1], c='y', label='Class 0')
    # Class 0이라고 예측한 데이터
    if tag == 'train':
        plt.scatter(support_vector[:, 0], support_vector[:, 1], c='r', s=150, alpha=0.3)
        # support vector
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")

    """hyperplane 시각화"""
    # plt.contour 검색해보기
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    x1range = np.linspace(x1_min, x1_max)
    x2range = np.linspace(x2_min, x2_max)
    X1, X2 = np.meshgrid(x1range, x2range)
    Z = np.empty(X1.shape)
    for (i, j), val in np.ndenumerate(X1):
        x1 = val
        x2 = X2[i, j]
        p = model.decision_function([[x1, x2]])
        Z[i, j] = p[0]
    if tag == 'train':
        levels = [-1, 0, 1]
        plt.contour(X1, X2, Z, levels, colors='k', linestyles=['dashed', 'solid', 'dashed'])
        # 결정경계(Z)는 Wx = 0, positive hyperplane은 Wx = +1, negative hyperplane은 Wx = -1
    else:
        levels = [0]
        plt.contour(X1, X2, Z, levels, colors='k', linestyles=['solid'])

def model_border(model, X, y, contour=True):
    xmin = X[:, 0].min()
    xmax = X[:, 0].max()
    ymin = X[:, 1].min()
    ymax = X[:, 1].max()
    xminmax = np.arange(xmin, xmax, .02)
    yminmax = np.arange(ymin, ymax, .02)
    X1, X2 = np.meshgrid(xminmax, yminmax)
    X_pred = np.c_[X1.ravel(), X2.ravel()]
    Z = model.predict(X_pred).reshape(X1.shape)
    #plt.figure(figsize = (10, 5))
    plt.contourf(X1, X2, Z, alpha=0.3)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "ro", alpha=1)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=1)
    plt.plot(X[:, 0][y==2], X[:, 1][y==2], "y^", alpha=1)
    plt.xlim([4.4, 7.6])
    plt.ylim([2, 4.0])
