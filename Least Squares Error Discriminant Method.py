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
"------------------提供维数修改---------------------"
# 加载鸢尾花数据集
iris = load_iris()
X = iris.data[:, :2]
y = iris.target

"------------------2、应用模型并训练------------------------"
"--------模型待展开----------"
# 对X和y使用Fisher's LDA 算法，训练好的模型保存在类属性里，使用predict调用
class LSEDA:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean_vectors = []
        self.cov_matrices = []
        for cls in self.classes:
            X_cls = X[y == cls]
            self.mean_vectors.append(np.mean(X_cls, axis=0))
            self.cov_matrices.append(np.cov(X_cls.T))

    def predict(self, X):
        predictions = []
        for x in X:
            min_distance = float('inf')
            predicted_class = None
            for cls, mean, cov in zip(self.classes, self.mean_vectors, self.cov_matrices):
                distance = np.linalg.norm(x - mean)
                if distance < min_distance:
                    min_distance = distance
                    predicted_class = cls
            predictions.append(predicted_class)
        return predictions


lseda = LSEDA()
lseda.fit(X, y)

"-------------------3、使用训练好的模型进行预测--------------------------"
predictions = lseda.predict(X)

"-------------------4、数据后处理-------------------------"
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

#散点图
def plot_decision_boundary(model, X, y):
    # Define grid range
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predict class labels for each point in the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array(Z).reshape(xx.shape)

    # Plot decision boundary and data points
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()

plot_decision_boundary(lseda, X, y)