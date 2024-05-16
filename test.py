import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay

# 加载鸢尾花数据集
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# 对X和y使用Fisher's LDA 算法，训练好的模型保存在类属性里，使用predict调用
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)

# 使用拟合出来的分割线对原始样本进行分割，此时返回的数据是类型判断数组，0，1，2分别代表属于什么类
# 结果统计
predictions = lda.predict(X)
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
"-----------------可视化------------------"
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

fig, ax = plt.subplots()
disp = DecisionBoundaryDisplay.from_estimator(
    lda,
    X,
    response_method="predict",
    cmap=plt.cm.coolwarm,
    alpha=0.8,
    ax=ax,
    xlabel=iris.feature_names[0],
    ylabel=iris.feature_names[1],
)

# 投射到直线
X_lda = lda.transform(X)
# 创建一个新的 DataFrame，包含主成分和对应的类别
df_lda = pd.DataFrame(data=X_lda, columns=['LD1', 'LD2'])
df_lda['Target'] = y

# 设置不同类别对应的颜色
colors = ['red', 'green', 'blue']

# 绘制散点图
plt.figure(figsize=(8, 6)) #绘制8*6图像

for i, target in enumerate(iris.target_names):
    plt.scatter(df_lda.loc[df_lda['Target'] == i, 'LD1'],
                df_lda.loc[df_lda['Target'] == i, 'LD2'],
                c=colors[i], label=target)