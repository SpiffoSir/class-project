import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as SklearnKNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.inspection import DecisionBoundaryDisplay
"使用两维向量训练模型"

"---------------------------------1、导入数据集---------------------------"
"------------------提供维数修改---------------------"
# 加载鸢尾花数据集
iris = load_iris()
X = iris.data[:, :2]
y = iris.target

"------------------2、应用模型并训练------------------------"
"--------模型待展开----------"
class KNeighborsClassifier:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.model = SklearnKNeighborsClassifier() #默认参数为 n_neighbors=3

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.model.fit(X, y)
    def predict(self, X):
        predictions = []
        for x_test in X:
            distances = []
            for x_train in self.X_train:
                distance = np.sqrt(np.sum((x_test - x_train) ** 2))
                distances.append(distance)
            nearest_neighbors = np.argsort(distances)[:self.n_neighbors]
            nearest_labels = [self.y_train[i] for i in nearest_neighbors]
            prediction = max(nearest_labels, key=nearest_labels.count)
            predictions.append(prediction)
        return predictions

knn = KNeighborsClassifier()



"-------------------3、使用训练好的模型进行预测--------------------------"
knn.fit(X, y)
predictions = knn.predict(X)

"-------------------4、数据后处理-------------------------"
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
"散点图和对应的决策边界"
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
    knn.model,
    X,
    response_method="predict",
    ax = ax,
    cmap = plt.cm.Paired
)
ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)
ax.set_title('Decision Boundary of SVC')
plt.show()