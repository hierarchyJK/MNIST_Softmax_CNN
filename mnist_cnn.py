# -*-coding:utf-8 -*-
"""
@project:untitled3
@author:Kun_J
@file:.py
@ide:untitled3
@time:2019-02-19 10:36:49
@month:二月
"""
"""加载数据集"""
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data('mnist/mnist.npz')

"""数据规范化"""
from keras import backend as K
#在C:\Users\korey\.keras\keras.json配置文件中查看默认的数据shape是channels_last
img_rows, img_cols = 28, 28
if K.image_data_format() == 'channels_first': #[batch, channel, height, width]
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:# channel_last  [batch, height, width, cbannel]
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# 数据类型转换float32
X_train = x_train.astype('float32')
X_test = x_test.astype('float32')
# 数据归一化
X_train /= 255
X_test /= 255

"""统计训练数据中各标签数量"""
import numpy as np
import matplotlib.pyplot as plt
label,count = np.unique(y_train, return_counts=True)

#可视化训练集标签数量
fig = plt.figure()
plt.bar(label, count, width=0.7, align='center' )
plt.title('Label Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(label)
plt.ylim(0, 7500)
for a, b in zip(label, count):
    plt.text(a, b, "%d" % b, ha = 'center', va = 'bottom', fontsize = 10)
plt.show()

"""数据处理：one-hot 编码"""
from keras.utils import np_utils

n_classes = 10
Y_train = np_utils.to_categorical(y_train, n_classes) # 感觉keras的封装太强大了
Y_test = np_utils.to_categorical(y_test, n_classes)

"""使用Keras sequential model定义MNIST CNN网络"""
from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
model = Sequential()

# Feature Extraction()
# 第一层：卷积，32个3*3卷积核，激活函数使用RELU
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                 input_shape=input_shape))
# 第二层：卷积，64个3*3卷积核，激活函数使用RELU
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
# 最大池化层，池化窗口2*2
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout 25%的输入神经元
model.add(Dropout(0.25))
# 将Pooled feature map 摊平输入全连接层
model.add(Flatten())

#Classification 全连接层
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
#使用softmax激活函数做多分类
model.add(Dense(n_classes, activation='softmax'))

"""查看MNIST CNN模型网络结构"""
model.summary()
for layer in model.layers:
    print(layer.get_output_at(0).get_shape().as_list())

"""编译模型"""
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
"""训练模型，并将指标保存到history中"""
history = model.fit(X_train, Y_train, batch_size=128, epochs=5, verbose=2, validation_data=(X_test, Y_test))

"""可视化指标"""
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'lower right')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc = 'upper right')
plt.show()

"""保存模型"""
import tensorflow as tf
import os
save_dir = './mnist/keras_cnn_model'
if tf.gfile.Exists(save_dir):
    tf.gfile.DeleteRecursively(save_dir)
tf.gfile.MakeDirs(save_dir)

model_name = 'keras_mnist.h5'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)

"""加载模型"""
from keras.models import load_model
mnist_cnn_model = load_model(model_path)

loss_and_metrics = mnist_cnn_model.evaluate(X_test, Y_test, verbose=2)
print("Test Loss: {}" .format(loss_and_metrics[0]))
print("Train Loss:{}%".format(loss_and_metrics[1] * 100))

predictes_classes = mnist_cnn_model.predict_classes(X_test)

correct_indices = np.nonzero(predictes_classes == y_test)[0]
incorrect_indices = np.nonzero(predictes_classes != y_test)[0]
print("Classified correctly count:{}".format(len(correct_indices)))
print("Classified incorrectly count:{}".format(len(incorrect_indices)))


















