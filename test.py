import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.inspection import DecisionBoundaryDisplay

# 生成一些示例数据
iris = datasets.load_iris()
np.random.seed(0)
class1_data = np.random.randn(50, 2) + np.array([2, 2])
class2_data = np.random.randn(50, 2) + np.array([5, 5])
X = iris.data[:, :2]


# 可视化数据
plt.scatter(class1_data[:,0], class1_data[:,1], color='blue', label='Class 1')
plt.scatter(class2_data[:,0], class2_data[:,1], color='red', label='Class 2')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Sample Data')
plt.legend()
plt.show()


print("ok")