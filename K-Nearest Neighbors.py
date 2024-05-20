import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
"使用两维向量训练模型"

"---------------------------------1、导入数据集---------------------------"
"------------------提供维数修改---------------------"
# 加载鸢尾花数据集
iris = load_iris()
X = iris.data[:, :2]
y = iris.target

"------------------2、应用模型并训练------------------------"
"--------模型待展开----------"
# 创建KNN分类器
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
# 创建一个网格来绘制决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

"-------------------3、使用训练好的模型进行预测--------------------------"
predictions = knn.predict(np.c_[xx.ravel(), yy.ravel()])
predictions = predictions.reshape(xx.shape)
"-------------------4、数据后处理-------------------------"
"此时由于prediction返回结果不再是类型所属的矩阵，所以此时混淆矩阵不能用返回参数进行计算，"
# "计算准确度和混淆矩阵"
# prediction_arr = np.array([0,0,0])
# y_arr = np.array([0,0,0])
# predictions_accuracy = 0
# count = 0
# wrong_list = []
# for i in predictions:
#     prediction_arr[i] += 1
# for i in y:
#     y_arr[i] += 1
# for i,j in zip(y,predictions):
#     count += 1
#     if i != j:
#         predictions_accuracy += 1
#         wrong_list.append(count)
#
# predictions_accuracy = (len(y) - predictions_accuracy)/len(y)
# print("y samples:")
# print(y_arr)
# print("predict result:")
# print(prediction_arr)
# print("predictions_accuracy:")
# print(predictions_accuracy)
# print("wrong_list:")
# print(wrong_list)

# # 计算混淆矩阵
# conf_matrix = confusion_matrix(y, predictions)
# print("conf_matrix:")
# print(conf_matrix)
"-----------------5、可视化------------------"
"散点图和对应的决策边界"
# 绘制决策边界
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, predictions, alpha=0.3, cmap=plt.cm.RdYlBu)
# 绘制训练数据点
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='RdYlBu', marker='o', s=100, label='Train')
# 绘制测试数据点
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='RdYlBu', marker='s', s=100, label='Test')
# 设置标题和标签
plt.title('KNN Classification Decision Boundary')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend()
plt.show()
