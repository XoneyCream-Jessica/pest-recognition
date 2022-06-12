import tensorflow.keras as keras
import os
import numpy as np
from tensorflow.keras import backend as keras_backend
import common.file as common_file
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# model_path = os.path.join(common_file.get_models_path(), "model_best.h5")
model_path = os.path.join(common_file.get_models_path(), "final")
model = keras.models.load_model(model_path, custom_objects={'keras_backend': keras_backend})

data_dir = os.path.join(common_file.get_data_path(), "41all")

test_img_path = os.path.join(data_dir, 'test')
im_height = 224
im_width = 224
batch_size = 1
# 定义验证集图像生成器，并对图像进行预处理
test_image_generator = ImageDataGenerator(rescale=1. / 255)

test_data_gen = test_image_generator.flow_from_directory(directory=test_img_path,
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         target_size=(im_height, im_width),
                                                         class_mode='categorical')  # one-hot编码
print(test_data_gen)

correct = 0
total = 0

for _ in range(test_data_gen.n):
    images_test, labels_test = test_data_gen.next()
    predict = model.predict(images_test)
    total += 1
    correct += (1 if np.argmax(predict) == np.argmax(labels_test) else 0)

print(correct / total)
