import numpy as np
import matplotlib.pyplot as plt

# 生成一些示例数据
np.random.seed(0)
class1_data = np.random.randn(50, 2) + np.array([2, 2])
class2_data = np.random.randn(50, 2) + np.array([5, 5])

# 可视化数据
plt.scatter(class1_data[:,0], class1_data[:,1], color='blue', label='Class 1')
plt.scatter(class2_data[:,0], class2_data[:,1], color='red', label='Class 2')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Sample Data')
plt.legend()
plt.show()

# 计算类内散度矩阵
mean_class1 = np.mean(class1_data, axis=0)
mean_class2 = np.mean(class2_data, axis=0)
within_class_scatter = np.dot((class1_data - mean_class1).T, (class1_data - mean_class1)) + np.dot((class2_data - mean_class2).T, (class2_data - mean_class2))

# 计算类间散度矩阵
between_class_scatter = np.dot((mean_class1 - mean_class2).reshape(-1, 1), (mean_class1 - mean_class2).reshape(1, -1))

# 计算Fisher判别准则
fisher_criteria = np.dot(np.linalg.inv(within_class_scatter), between_class_scatter)

# 计算最优判别方向（特征向量）
eigen_values, eigen_vectors = np.linalg.eig(fisher_criteria)
optimal_direction = eigen_vectors[:, np.argmax(eigen_values)]

# 投影数据到最优判别方向上
projected_class1_data = np.dot(class1_data, optimal_direction)
projected_class2_data = np.dot(class2_data, optimal_direction)

# 可视化投影后的数据
plt.hist(projected_class1_data, color='blue', alpha=0.5, label='Class 1', bins=10)
plt.hist(projected_class2_data, color='red', alpha=0.5, label='Class 2', bins=10)
plt.xlabel('Projection on Optimal Direction')
plt.ylabel('Frequency')
plt.title('Projected Data')
plt.legend()
plt.show()



'隔离'
# 计算最优判别直线的斜率和截距
slope = optimal_direction[1] / optimal_direction[0]
intercept = mean_class1[1] - slope * mean_class1[0]

# 绘制原始数据点和最优判别直线
plt.scatter(class1_data[:,0], class1_data[:,1], color='blue', label='Class 1')
plt.scatter(class2_data[:,0], class2_data[:,1], color='red', label='Class 2')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 绘制最优判别直线
x_vals = np.array(plt.gca().get_xlim())
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, color='green', linestyle='--', label='Optimal Discriminant Line')

plt.title('Sample Data with Optimal Discriminant Line')
plt.legend()
plt.show()
