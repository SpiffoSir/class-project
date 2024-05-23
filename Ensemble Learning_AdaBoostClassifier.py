from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.base import clone
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
"--------模型待展开----------"
# 使用决策树桩作为基分类器
class AdaBoostClassifier:
    def __init__(self, base_estimator='decision_tree', n_estimators=10, random_state=None):
        self.base_estimator = DecisionTreeClassifier(
            max_depth=1) if base_estimator == 'decision_tree' else base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Initialize weights
        sample_weights = np.ones(n_samples) / n_samples

        for iboost in range(self.n_estimators):
            # Clone the base estimator
            estimator = clone(self.base_estimator)

            # Fit the estimator
            estimator.fit(X, y, sample_weight=sample_weights)

            # Predict on the training data
            y_pred = estimator.predict(X)

            # Compute the indicator function
            incorrect = (y_pred != y)

            # Calculate estimator error
            estimator_error = np.dot(sample_weights, incorrect) / np.sum(sample_weights)

            # If the error is 0 or the error is greater than or equal to 0.5, break
            if estimator_error >= 0.5:
                break
            if estimator_error == 0:
                estimator_weight = 1
            else:
                # Calculate estimator weight
                estimator_weight = np.log((1. - estimator_error) / max(estimator_error, 1e-10))

            # Store the current estimator and its weight
            self.estimators_.append(estimator)
            self.estimator_weights_.append(estimator_weight)
            self.estimator_errors_.append(estimator_error)

            # Update sample weights
            sample_weights *= np.exp(estimator_weight * incorrect * ((sample_weights > 0) | (estimator_weight < 0)))

            # Normalize the sample weights
            sample_weights /= np.sum(sample_weights)

    def predict(self, X):
        # Aggregate predictions from all estimators
        pred = np.zeros((X.shape[0], self.n_classes_))

        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            pred += weight * self._one_hot_encode(estimator.predict(X))

        # Return the class with the highest aggregated weight
        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def _one_hot_encode(self, y):
        one_hot = np.zeros((y.shape[0], self.n_classes_))
        for idx, class_ in enumerate(self.classes_):
            one_hot[y == class_, idx] = 1
        return one_hot


ada_clf = AdaBoostClassifier(n_estimators=50, random_state=42)
ada_clf.fit(X, y)

"---------------------------------3、训练模型并进行预测---------------------"
predictions = ada_clf.predict(X)



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
    ada_clf,
    X,
    response_method="predict",
    ax = ax,
    cmap = plt.cm.Paired
)
ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)
ax.set_title('Decision Boundary of SVC')
plt.show()
