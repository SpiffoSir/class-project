import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 加载模型
loaded_model = load_model('ann_test_1.h5')

# 预测并显示五个图像及其预测结果
num_images = 5
random_indices = np.random.randint(0, len(x_test), size=num_images)

plt.figure(figsize=(15, 3))
correct_predictions = 0
total_predictions = 5

for i, random_index in enumerate(random_indices):
    random_image = x_test[random_index]
    random_image_expanded = np.expand_dims(random_image, axis=0)
    predictions = loaded_model.predict(random_image_expanded)
    predicted_class = np.argmax(predictions)
    true_class = y_test[random_index]
    print(true_class)
    print(predicted_class)
    if predicted_class == true_class:
        correct_predictions += 1
        print("finish")
    # 在图像子图中显示
    plt.subplot(1, num_images, i + 1)
    plt.imshow(random_image, cmap='gray')
    plt.title(f'Predicted: {predicted_class}')
    plt.axis('off')

accuracy = correct_predictions / total_predictions
print(accuracy)
plt.show()

