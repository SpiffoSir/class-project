import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from sklearn.inspection import DecisionBoundaryDisplay
import seaborn as sns



"---------------------------------1、导入数据集---------------------------"
"------------------提供维数修改---------------------"
# 加载鸢尾花数据集
iris = load_iris()
X = iris.data[:, :2]
y = iris.target



"---------------------------------2、赋值数学属性------------------------"
class LinearDiscriminantAnalysis:
    def fit(self, X, y):
        # 计算每个类的均值向量
        self.means_ = []
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            self.means_.append(np.mean(X[y == cls], axis=0))
        self.means_ = np.array(self.means_)

        # 计算类内散布矩阵
        n_features = X.shape[1]
        self.Sw = np.zeros((n_features, n_features))
        for cls in self.classes_:
            Xi = X[y == cls]
            mean_vec = self.means_[cls].reshape(n_features, 1)
            scatter = np.dot((Xi - mean_vec.T).T, (Xi - mean_vec.T))
            self.Sw += scatter

        # 计算类间散布矩阵
        overall_mean = np.mean(X, axis=0).reshape(n_features, 1)
        self.Sb = np.zeros((n_features, n_features))
        for i, mean_vec in enumerate(self.means_):
            n = X[y == i, :].shape[0]
            mean_vec = mean_vec.reshape(n_features, 1)
            scatter = n * np.dot((mean_vec - overall_mean), (mean_vec - overall_mean).T)
            self.Sb += scatter

        # 计算Sw^-1 * Sb 的特征值和特征向量
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(self.Sw).dot(self.Sb))
        eig_pairs = [(eigvals[i], eigvecs[:, i]) for i in range(len(eigvals))]
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

        # 选择最多特征向量，降维
        self.W = np.hstack([eig_pairs[i][1].reshape(n_features, 1) for i in range(len(self.classes_) - 1)])

        return self

    def predict(self, X):
        X_lda = X.dot(self.W)
        means_lda = self.means_.dot(self.W)

        # 分类
        y_pred = []
        for x in X_lda:
            distances = [np.linalg.norm(x - mean_lda) for mean_lda in means_lda]
            y_pred.append(np.argmin(distances))
        return np.array(y_pred)

lda = LinearDiscriminantAnalysis()



"---------------------------------3、训练模型并进行预测----------------"
lda.fit(X, y)
predictions = lda.predict(X)



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
#决策边界
fig, ax = plt.subplots()
disp = DecisionBoundaryDisplay.from_estimator(
    lda,
    X,
    response_method="predict",
    ax = ax,
    cmap = plt.cm.Paired
)
ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)
ax.set_title('Decision Boundary of SVC')
plt.show()

