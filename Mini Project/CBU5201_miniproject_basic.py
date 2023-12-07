import tensorflow as tf
import keras
import sklearn
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import datetime
import glob
import os

# 设置GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU') # 设置GPU为可见
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
       print(e) # 打印异常


# 加载GENKI数据集
def load_genki_data():
    images = []
    labels = []
    poses = []

    with open('./Mini Project/genki4k/labels.txt', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            img_path = f'./Mini Project/genki4k/files/file{i+1:04}.jpg'
            img = cv2.imread(img_path)
            img = cv2.resize(img, (256, 256))
            images.append(img)

            # 解析标签信息
            label, yaw, pitch, roll = map(float, line.strip().split())
            labels.append(int(label))
            poses.append([yaw, pitch, roll])

    images = np.array(images)
    labels = keras.utils.to_categorical(labels, num_classes=2)  # one-hot编码笑脸标签
    poses = np.array(poses)

    return images, labels, poses

# 划分训练集和测试集
def split_data(images, labels, test_size=0.2, random_state=42):
    return sklearn.model_selection.train_test_split(images, labels, test_size=test_size, random_state=random_state)

# 构建笑脸分类模型
def build_model(input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 加载模型
# model_files = glob.glob('./Mini Project/model_*.h5') # 获取所有模型文件
# latest_model_file = max(model_files, key=os.path.getctime) # 找到最新的模型文件
# model = keras.models.load_model(latest_model_file) # 加载最新的模型

# 加载数据集
images, labels, poses= load_genki_data()

# 划分训练集和测试集
train_images, test_images, train_labels, test_labels = split_data(images, labels)

# 检查训练集和测试集中是否有 None 值
print(np.any(train_images == None))
print(np.any(test_images == None))
print(np.any(train_labels == None))
print(np.any(test_labels == None))

# 构建并训练模型
input_shape = (256, 256, 3)
model = build_model(input_shape)
model.fit(train_images, train_labels, epochs=5, batch_size=32, validation_data=(test_images, test_labels))

# 保存模型
save_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
model.save(f'./Mini Project/model_{save_time}.h5')

