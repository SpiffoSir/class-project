from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import clone
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay
import seaborn as sns
import matplotlib.pyplot as plt
"使用两维向量训练模型"

"---------------------------------1、导入数据集---------------------------"
"------------------提供维数修改---------------------"
# 加载鸢尾花数据集
iris = load_iris()
X = iris.data[:, :2]
y = iris.target

"---------------------------------2、赋值数学属性--------------------------"
# 使用决策树桩作为基分类器
class BaggingClassifier:
    def __init__(self, base_estimator='decision_tree', n_estimators=10, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.base_estimator_type = base_estimator
        self.estimators_ = []

        # 根据字符串参数选择基分类器
        if base_estimator == 'decision_tree':
            self.base_estimator = DecisionTreeClassifier()
        elif base_estimator == 'knn':
            self.base_estimator = KNeighborsClassifier()
        elif base_estimator == 'lda':
            self.base_estimator = LinearDiscriminantAnalysis()
        elif base_estimator == 'svc':
            self.base_estimator = SVC(probability=True)
        else:
            raise ValueError(f"Unsupported base_estimator type: {base_estimator}")

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            # 随机采样样本
            indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # 克隆基学习器并训练
            estimator = clone(self.base_estimator)
            estimator.fit(X_bootstrap, y_bootstrap)
            self.estimators_.append(estimator)

    def predict(self, X):
        # 对每个基学习器进行预测
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])
        # 对预测结果进行投票
        y_pred = np.mean(predictions, axis=0)
        return np.round(y_pred).astype(int)

"decision_tree,knn,lad,svc随便选一个"
bagging_clf = BaggingClassifier(base_estimator='decision_tree', n_estimators=50, random_state=42)


"---------------------------------3、训练模型并进行预测---------------------"
bagging_clf.fit(X, y)
predictions = bagging_clf.predict(X)



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
    bagging_clf,
    X,
    response_method="predict",
    ax = ax,
    cmap = plt.cm.Paired
)
ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)
ax.set_title('Decision Boundary of SVC')
plt.show()
