import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.models import load_model

# 加载数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
loaded_model = load_model('ann_test_1.h5')

# 保存模型
loaded_model.save('ann_test_1.h5')

# 预测
random_index = np.random.randint(0, len(x_test))
random_image = x_test[random_index]
random_image_expanded = np.expand_dims(random_image, axis=0)
predictions = loaded_model.predict(random_image_expanded)
predicted_class = np.argmax(predictions)
print(f'Predicted class: {predicted_class}')
