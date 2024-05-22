import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.models import load_model

# 加载数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#loaded_model = load_model('ann_test_1.h5')
# 归一化,压缩灰度图为0-1范围小数
x_train = x_train / 255.0
x_test = x_test / 255.0

# 制作类别向量保存待分类的内容
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 定义模型,直接使用keras库构建模型
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
# 保存模型
model.save('ann_test_1.h5')


# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# 预测
random_index = np.random.randint(0, len(x_test))
random_image = x_test[random_index]
random_image_expanded = np.expand_dims(random_image, axis=0)
predictions = model.predict(random_image_expanded)
predicted_class = np.argmax(predictions)
print(f'Predicted class: {predicted_class}')
