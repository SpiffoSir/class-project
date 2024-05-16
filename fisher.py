import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 加载鸢尾花数据集
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# 拟合分割线
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)

# 使用拟合出来的分割线对原始样本进行分割
predictions = lda.predict(X)

"-----------------可视化------------------"
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

# 绘制决策边界（拟合出来的直线）
for i in range(len(iris.target_names) - 1):  # 循环遍历类别
    coef = lda.coef_[i]  # 取对应类别的权重向量
    intercept = lda.intercept_[i]  # 取对应类别的截距

    # 画出决策边界直线
    plt.plot([X_lda[:, 0].min(), X_lda[:, 0].max()],
             [-(coef[0]*X_lda[:, 0].min() + intercept)/coef[1],
              -(coef[0]*X_lda[:, 0].max() + intercept)/coef[1]],
             color='black', linestyle='--')

plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('Linear Discriminant Analysis of Iris Dataset')
plt.legend()
plt.grid(True)
plt.show()