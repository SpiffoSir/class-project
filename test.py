from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 使用决策树桩作为基分类器
base_clf = DecisionTreeClassifier(max_depth=1)

# 1. BaggingClassifier
bagging_clf = BaggingClassifier(estimator=base_clf, n_estimators=50, random_state=42)
bagging_clf.fit(X_train, y_train)
y_pred_bagging = bagging_clf.predict(X_test)
accuracy_bagging = accuracy_score(y_test, y_pred_bagging)
print(f'BaggingClassifier Accuracy: {accuracy_bagging}')

# 2. RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'RandomForestClassifier Accuracy: {accuracy_rf}')

# 3. AdaBoostClassifier
ada_clf = AdaBoostClassifier(estimator=base_clf, n_estimators=50, algorithm='SAMME', random_state=42)
ada_clf.fit(X_train, y_train)
y_pred_ada = ada_clf.predict(X_test)
accuracy_ada = accuracy_score(y_test, y_pred_ada)
print(f'AdaBoostClassifier Accuracy: {accuracy_ada}')
