# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import datetime

import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
physical_devices = tf.config.experimental.list_physical_devices('CPU') \
    if (gpu_devices is None) or (len(gpu_devices) == 0) else gpu_devices
assert len(physical_devices) > 0, "Not enough GPU/CPU hardware devices available"
if (gpu_devices is not None) and (len(gpu_devices) > 0):
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
# import pandas as pd
import os
import time
# bufrom tensorflow_core.python.keras.models import load_model
import common.file as common_file

im_height = 224
im_width = 224
batch_size = 32
epochs = 40
classnum = 41

data_dir = os.path.join(common_file.get_data_path(), "41all")
train_dir = os.path.join(data_dir, "train")  # 训练集路径
validation_dir = os.path.join(data_dir, "val")  # 验证集路径

save_log_image_path = common_file.get_log_path()  # 保存路径

# 定义训练集图像生成器，并进行图像增强
train_image_generator = ImageDataGenerator(rescale=1. / 255,  # 归一化
                                           rotation_range=30,  # 旋转范围
                                           width_shift_range=0.2,  # 水平平移范围
                                           height_shift_range=0.2,  # 垂直平移范围
                                           shear_range=0.2,  # 剪切变换的程度
                                           zoom_range=0.2,  # 剪切变换的程度
                                           horizontal_flip=True,  # 水平翻转
                                           vertical_flip=True,  # 竖直翻转
                                           fill_mode='nearest'
                                           )

# 使用图像生成器从文件夹train_dir中读取样本，对标签进行one-hot编码
train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,  # 从训练集路径读取图片
                                                           batch_size=batch_size,  # 一次训练所选取的样本数
                                                           shuffle=True,  # 打乱标签
                                                           target_size=(im_height, im_width),  # 图片resize到224x224大小
                                                           class_mode='categorical')  # one-hot编码

# 训练集样本数
total_train = train_data_gen.n

# 定义验证集图像生成器，并对图像进行预处理
validation_image_generator = ImageDataGenerator(rescale=1. / 255,  # 归一化
                                                )

# 使用图像生成器从验证集validation_dir中读取样本
val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,  # 从验证集路径读取图片
                                                              batch_size=batch_size,  # 一次训练所选取的样本数
                                                              shuffle=False,  # 不打乱标签
                                                              target_size=(im_height, im_width),  # 图片resize到224x224大小
                                                              class_mode='categorical')  # one-hot编码

# 验证集样本数
total_val = val_data_gen.n

# !----------------------------------------------------------------------------------------------------------------

# 使用tf.keras.applications中的InceptionV3网络，并且使用官方的预训练模型
inputs = tf.keras.layers.Input(shape=(224, 224, 3))
covn_base = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False,
                                              input_tensor=inputs)  # ResNet50V2.  .MobileNetV2
covn_base.trainable = True

# 冻结前面的层，训练最后40层
for layers in covn_base.layers[:-100]:
    layers.trainable = False

# # 构建模型 初始化模型
now_time = time.time()
net = tf.keras.layers.GlobalAveragePooling2D()(covn_base.layers[-1].output)
net = tf.keras.layers.Dense(classnum, activation='softmax')(net)
model = tf.keras.models.Model(inputs=inputs, outputs=net)
model.summary()  # 打印每层参数信息

# # # 构建模型 初始化模型
# model = tf.keras.Sequential()
# model.add(covn_base)
# model.add(tf.keras.layers.GlobalAveragePooling2D())  # 加入全局平均池化层
# # model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(1024, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.2))
# model.add(tf.keras.layers.Dense(classnum, activation='softmax'))  # 加入输出层
# model.summary()  # 打印每层参数信息

# 继续训练
# -------------------------------------------------------------------------------------------------------------------
# model = tf.keras.models.load_model("J:\\insecttest\\finaldata\\model1\\finalh5\\path_to_my_model.h5")
# model.summary()  # 打印每层参数信息
# -------------------------------------------------------------------------------------------------------------------

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # 使用adam优化器，学习率为0.0001  0.0004
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),  # 交叉熵损失函数
              metrics=["accuracy"])  # 评价函数

# 保存pb
# ------------------------------------------------------------------------------------------------------------------------
# 设置回调，每一个epoch都保存一次结果
model_each_epoch_save_callback = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(common_file.get_models_path(), 'model_{epoch:03d}'),  # 路径修改
    save_best_only=True, mode='auto',
    save_freq='epoch')
# 保存最优模型
model_best_callback = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(common_file.get_models_path(), 'model_best.h5'),
    save_best_only=True, monitor='val_accuracy', verbose=2
)
# ------------------------------------------------------------------------------------------------------------------------

# 保存log
# ------------------------------------------------------------------------------------------------------------------------
stamp = datetime.datetime.now().strftime("Y%m%d-%H%M%S")
logdir = os.path.join(common_file.get_log_path(), "pest_recognition." + stamp)  # 路径修改
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

# 开始训练
# -----------------------------------------------------------------------------------------------------------------------
history = model.fit(x=train_data_gen,  # 输入训练集
                    steps_per_epoch=total_train // batch_size,  # 一个epoch包含的训练步数
                    epochs=epochs,  # 训练模型迭代次数
                    validation_data=val_data_gen,  # 输入验证集
                    validation_steps=total_val // batch_size,  # 一个epoch包含的训练步数
                    callbacks=[tensorboard_callback, model_each_epoch_save_callback, model_best_callback]
                    )

# 记录训练集和验证集的准确率和损失值
history_dict = history.history
train_loss = history_dict["loss"]  # 训练集损失值
train_accuracy = history_dict["accuracy"]  # 训练集准确率
val_loss = history_dict["val_loss"]  # 验证集损失值
val_accuracy = history_dict["val_accuracy"]  # 验证集准确率
total_time = time.time() - now_time
print("total_time", total_time)

# 保存
# -------------------------------------------------------------------------------------
# 保存模型pb
saved_model_path = os.path.join(common_file.get_models_path(), "final")  # 路径修改
if not (os.path.isdir(saved_model_path)):
    os.mkdir(saved_model_path)
model.save(saved_model_path)

# 保存模型h5
saved_model_path = os.path.join(common_file.get_models_path(), "finalh5")
if not (os.path.isdir(saved_model_path)):
    os.mkdir(saved_model_path)
model.save(os.path.join(saved_model_path, 'path_to_my_model.h5'))  # 路径修改
# -------------------------------------------------------------------------------------


plt.figure()
plt.plot(range(epochs), train_loss, label='train_loss')
plt.plot(range(epochs), val_loss, label='val_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig(save_log_image_path + "\\" + "loss.png")

plt.figure()
plt.plot(range(epochs), train_accuracy, label='train_accuracy')
plt.plot(range(epochs), val_accuracy, label='val_accuracy')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.savefig(save_log_image_path + "\\" + "accuracy.png")

plt.show()

# —————————————————————————————————————————————————————————————————————————————————————————————————————————————————
