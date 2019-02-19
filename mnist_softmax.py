# -*-coding:utf-8 -*-
"""
@project:untitled3
@author:Kun_J
@file:.py
@ide:untitled3
@time:2019-02-14 15:20:36
@month:二月
"""
#使用tf.contrib.learn模块加载MNIST数据集(Deprecated 弃用)
#import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('./mnist/dataset/') 这种方法官方已经遗弃了
"""使用keras加载MNIST数据集,比上面的方法方便很多
   tf.keras.datasets.mnist.load_data(path = 'mnist.npz')
   Arguments:
             path:本地缓存MNIST数据集（mnist.npz）的相对路径（~/.keras/datasets）
"""
from keras.datasets import mnist
(x_train, y_train), (x_test,y_test) = mnist.load_data('mnist/mnist.npz')## 这里是相对路径，其实绝对路径在这里哦C:\Users\korey\.keras\datasets\mnist\mnist.npz
print(x_train.shape, y_train.shape, type(x_train)) #(60000, 28, 28) (60000,)
print(x_test.shape, x_test.shape )#(10000, 28, 28) (10000, 28, 28)

"""MNSIT数据集 样例可视化"""
import matplotlib.pyplot as plt
fig = plt.figure() #先绘制一张空白图
for i in range(15):
    plt.subplot(3, 5,(i+1)) #绘制前15个子图，以3行5列子图形式展示
    plt.imshow(x_train[i], cmap="Greys") #使用灰色显示灰度值
    plt.title("Label{}".format(y_train[i])) #设置标签为子图标签
    plt.xticks([]) # 删除x标记
    plt.yticks([]) # 删除y标记
##print(x_train[0])

"""下面开始搭建MNIST的softmax网络
   第一步 数据处理：规范化
"""
X_train = x_train.reshape(60000, 784) # （60000,784）
X_test = x_test.reshape(10000, 784)#（10000,784）
print(X_train.shape, type(X_train))
print(X_test.shape, type(X_test))
#将数据类型装换为float32,如果不怎么做的话，后面的归一化操作得到的只有0和1两个数
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#数据归一化
X_train /= 255
X_test /= 255

"""统计数据的中各种标签的数量"""
import numpy as np
label, count = np.unique(y_train, return_counts=True)
print(label, count)#[0 1 2 3 4 5 6 7 8 9] [5923 6742 5958 6131 5842 5421 5918 6265 5851 5949]
fig = plt.figure()
plt.bar(label, count, width = 0.7, align = "center")
plt.title("Label Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(label)
plt.ylim(0,7500)

for a,b in zip(label, count):
    plt.text(a, b, "%d"% b, ha="center",va='bottom', fontsize = 10)
plt.show()

#one-hot
from keras.utils import np_utils
n_classes =10
print("Shape before one-hot encoding:", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes) ## 实现one-hot编码
print("Shape after one-hot encoding:", Y_train.shape)

Y_test = np_utils.to_categorical(y_test, n_classes)

print(y_train[0]) # 5
print(Y_train[0]) #[0,0,0,0,1,0,0,0,0,]

"""这里我们采用keras中的Sequential模型"""
from keras.models import Sequential
from keras.layers.core import Dense, Activation
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))

model.add(Dense(512)) ##这里应为是Sequential模型，故可以不用加input_shape()
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])# 编译模型
history = model.fit(X_train, Y_train, batch_size=128, epochs=5, verbose=2, validation_data=(X_test, Y_test))# 训练模型，信息保存在在history

fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['trian','test'], loc = 'lower right')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper right')

plt.show()
"""保存模型"""
import os
import tensorflow as tf
save_dir = './mnist/model'
if tf.gfile.Exists(save_dir):##每次保存判断模型是不是存在的话，则删除重新保存训练的的模型
    tf.gfile.DeleteRecursively(save_dir)###递归删除目录下的的文件
tf.gfile.MakeDirs(save_dir)


model_name = "keras_mnist.h5"###模型时HDF5格式的也就是.h5文件后缀
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print("Saved train model at %s" % model_path)
"""加载模型"""
from keras.models import load_model
mnist_model = load_model(model_path)
loss_and_metrics = mnist_model.evaluate(X_test, Y_test, verbose = 2)
print("Test Loss:{}".format(loss_and_metrics[0]))
print("Test Accuracy:{}%".format(loss_and_metrics[1]*100))
predicted_classes = mnist_model.predict_classes(X_test)
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
print("Classified correctly count:{}".format(len(correct_indices)))
print("Clssified incorrected count:{}".format(len(incorrect_indices)))