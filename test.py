import pandas as pd
import numpy as np
import warnings
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.utils.multiclass import check_classification_targets, unique_labels
import seaborn as sns
import matplotlib.pyplot as plt
"使用两维向量训练模型"

"---------------------------------1、导入数据集---------------------------"
"------------------提供维数修改---------------------"
# 加载鸢尾花数据集
iris = load_iris()
X = iris.data[:, :2]
y = iris.target



"---------------------------------2、实例化类属性------------------------"
"--------模型待展开----------"
# 对X和y使用Fisher's LDA 算法，训练好的模型保存在类属性里，使用predict调用
lda = LinearDiscriminantAnalysis()



"---------------------------------3、训练模型并进行预测----------------"


def fit(self, X, y):
    # 制作类别向量
    self.classes_, y = np.unique(y, return_inverse=True)

    # 获取样本数和特征数
    n_samples, n_features = X.shape

    # 获取类别数
    n_classes = len(self.classes_)
    # 设置类别的先验概率
    if self.priors is None:
        self.priors_ = np.bincount(y) / float(n_samples)
    else:
        self.priors_ = np.array(self.priors)

    # 初始化协方差矩阵相关的变量
    cov = None
    store_covariance = self.store_covariance
    if store_covariance:
        cov = []

    # 初始化均值、尺度和旋转矩阵列表
    means = []
    scalings = []
    rotations = []

    # 对每个类别进行计算
    for ind in range(n_classes):
        # 获取属于当前类别的样本
        Xg = X[y == ind, :]

        # 计算当前类别的均值向量
        meang = Xg.mean(0)
        means.append(meang)

        # 如果某个类别只有一个样本，抛出错误
        if len(Xg) == 1:
            raise ValueError("y has only 1 sample in class %s, covariance is ill defined." % str(self.classes_[ind]))

        # 中心化数据
        Xgc = Xg - meang

        # 进行SVD分解
        _, S, Vt = np.linalg.svd(Xgc, full_matrices=False)

        # 计算有效特征数（秩）
        rank = np.sum(S > self.tol)
        if rank < n_features:
            warnings.warn("Variables are collinear")

        # 计算方差并调整
        S2 = (S ** 2) / (len(Xg) - 1)
        S2 = ((1 - self.reg_param) * S2) + self.reg_param

        # 如果需要存储协方差矩阵
        if self.store_covariance or store_covariance:
            cov.append(np.dot(S2 * Vt.T, Vt))

        # 存储尺度和旋转矩阵
        scalings.append(S2)
        rotations.append(Vt.T)

    # 存储协方差矩阵
    if self.store_covariance or store_covariance:
        self.covariance_ = cov

    # 存储均值、尺度和旋转矩阵
    self.means_ = np.asarray(means)
    self.scalings_ = scalings
    self.rotations_ = rotations

    return self

lda.fit(X, y)

"预测"
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

