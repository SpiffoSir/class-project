import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

"------------------2、应用模型并训练------------------------"
"--------模型待展开----------"
# 对X和y使用Fisher's LDA 算法，训练好的模型保存在类属性里，使用predict调用
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)

"-------------------3、使用训练好的模型进行预测--------------------------"
predictions = lda.predict(X)

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
# # 投射到直线
# X_lda = lda.transform(X)
# # 创建一个新的 DataFrame，包含主成分和对应的类别
# df_lda = pd.DataFrame(data=X_lda, columns=['LD1', 'LD2'])
# df_lda['Target'] = y
# # 设置不同类别对应的颜色
# colors = ['red', 'green', 'blue']
# # 绘制散点图
# plt.figure(figsize=(8, 6)) #绘制8*6图像
# for i, target in enumerate(iris.target_names):
#     plt.scatter(df_lda.loc[df_lda['Target'] == i, 'LD1'],
#                 df_lda.loc[df_lda['Target'] == i, 'LD2'],
#                 c=colors[i], label=target)
# # 绘制决策边界
# for i in range(len(iris.target_names) - 1):  # 循环遍历类别
#     coef = lda.coef_[i]  # 取对应类别的权重向量
#     intercept = lda.intercept_[i]  # 取对应类别的截距
#     # 画出决策边界直线
#     plt.plot([X_lda[:, 0].min(), X_lda[:, 0].max()],
#              [-(coef[0]*X_lda[:, 0].min() + intercept)/coef[1],
#               -(coef[0]*X_lda[:, 0].max() + intercept)/coef[1]],
#              color='black', linestyle='--')
# plt.xlabel('LD1')
# plt.ylabel('LD2')
# plt.title('Linear Discriminant Analysis of Iris Dataset')
# plt.legend()
# plt.grid(True)
# plt.show()
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