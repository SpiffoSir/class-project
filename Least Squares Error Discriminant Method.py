import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from sklearn.inspection import DecisionBoundaryDisplay
import seaborn as sns

"使用两维向量训练模型"

"---------------------------------1、导入数据集---------------------------"
"------------------提供维数修改,现在调整为二分类---------------------"
# 加载鸢尾花数据集
iris = load_iris()
X = iris.data[:, :2]
y = iris.target

mask = y < 2
X = X[mask]
y = y[mask]
"---------------------------------2、赋值数学属性------------------------"
"--------模型待展开----------"


class LSEDA:
    def fit(self, X, y):
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        # 计算权重
        X_transpose = X.T
        self.weights_ = np.linalg.inv(X_transpose @ X) @ X_transpose @ y

    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        # 计算预测值
        y_pred = X @ self.weights_

        # 将预测值转换为类别标签（这里假设二分类问题，使用0.5作为阈值）
        y_pred_class = (y_pred >= 0.5).astype(int)

        return y_pred_class


lseda = LSEDA()


"---------------------------------3、训练模型并进行预测----------------"
lseda.fit(X, y)
predictions = lseda.predict(X)


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
"-----------------5、可视化------------------"
"包含散点和混淆矩阵使用"
#混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

#决策边界
fig, ax = plt.subplots()
disp = DecisionBoundaryDisplay.from_estimator(
    lseda,
    X,
    response_method="predict",
    ax = ax,
    cmap=plt.cm.Paired)
ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)
ax.set_title('Decision Boundary of SVC')
plt.show()