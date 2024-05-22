import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC as SklearnSVC
from sklearn.metrics import confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay
import seaborn as sns


"---------------------------------1、导入数据集---------------------------"
"------------------提供维数修改---------------------"
# 加载鸢尾花数据集
iris = load_iris()
X = iris.data[:, :2]
y = iris.target

"---------------------------------2、赋值数学属性------------------------"
"--------模型待展开----------"

class SVC:
    def __init__(self, kernel='linear', C=1.0):
        self.kernel = kernel
        self.C = C
        self.model = SklearnSVC(kernel=self.kernel, C=self.C)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

svm = SVC()  #可选项，设定默认为linear,c = 1


"---------------------------------3、训练模型并进行预测----------------"
svm.fit(X, y)
predictions = svm.predict(X)

"---------------------------------4、数据后处理----------------------------"
"计算准确度和混淆矩阵"
prediction_arr = np.array([0,0,0])
y_arr = np.array([0,0,0])
predictions_accuracy = 0
count = 0
wrong_list = []
for i in predictions:
    prediction_arr[i] += 1
for i in y:
    y_arr[i] += 1
for i,j in zip(y,predictions):
    count += 1
    if i != j:
        predictions_accuracy += 1
        wrong_list.append(count)

predictions_accuracy = (len(y) - predictions_accuracy)/len(y)
print("y samples:")
print(y_arr)
print("predict result:")
print(prediction_arr)
print("predictions_accuracy:")
print(predictions_accuracy)
print("wrong_list:")
print(wrong_list)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y, predictions)
print("conf_matrix:")
print(conf_matrix)
"----------------------------------5、可视化-----------------------------"
"包含散点和混淆矩阵使用"

#混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

fig, ax = plt.subplots()
disp = DecisionBoundaryDisplay.from_estimator(
    svm.model,
    X,
    response_method="predict",
    ax = ax,
    cmap=plt.cm.Paired)
ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)
ax.set_title('Decision Boundary of SVC')
plt.show()