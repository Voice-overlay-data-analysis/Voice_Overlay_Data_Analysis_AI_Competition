from matplotlib import pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import export_graphviz
import graphviz
import pydot
from module import *

iris = load_iris()
keys = iris.keys()
X = iris['data']
Y = iris['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=0).fit(X_train, Y_train)
#Decision Tree 노드 결정 기준 : Information Gain (entropy, Gini)
#max_depth : 트리의 최대깊이를 제한함으로써 Overfitting 조절
#min_sample_split : 분할되기 위해 노드가 가져야할 최소 샘플 수 (default : 2)
#max_leaf_node : leaf node(트리의 마지막 노드)의 최대 수

export_graphviz(dt_model, out_file='./tree.dot', class_names=iris['target_names'], feature_names=iris['feature_names'], filled=True)
#Decision Tree 시각화
#.dot 파일로 저장
(graph,) = pydot.graph_from_dot_file('./tree.dot', encoding='utf8')
graph.write_png('./tree.png')
#.dot 파일을 읽어와서 .png 파일로 저장

print("* train scroe : {}".format(dt_model.score(X_train, Y_train)))
print("* test score : {}".format(dt_model.score(X_test, Y_test)))

importance = dt_model.feature_importances_
#y 결정에 대한 x_feature의 중요도