import os

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import models
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import common.file as common_file
import common.torch_util as torch_util
import common.plot_util as plot_util
import numpy as np

data_dir = os.path.join(common_file.get_data_path(), "41all")
im_height = 224
im_width = 224
batch_size = 1
device = torch_util.get_device()

print(device)

test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'test'),
                                                transform=transforms.Compose(
                                                    [
                                                        transforms.Resize((im_height, im_width)),
                                                        transforms.ToTensor()
                                                    ]))

test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = models.resnet50(pretrained=False)

class_names = test_dataset.classes

# 全连接层的输入通道in_channels个数
num_fc_in = model.fc.in_features

# 改变全连接层，41分类问题，out_features = 41
model.fc = torch.nn.Sequential(torch.nn.Dropout(0),
                               torch.nn.Linear(num_fc_in, 41))

model.load_state_dict(torch.load(os.path.join(common_file.get_models_path(), "pest_41.pth")))
model = model.to(device)
model.eval()

predict_arr = []
actual_arr = []
for images_test, labels_test in test_dataloader:
    images_test = images_test.to(device)
    labels_test = labels_test.to(device)
    outputs_test = model(images_test)
    predict_label = outputs_test.argmax(dim=-1)
    predict_arr.append(class_names[predict_label.item()])
    actual_arr.append(class_names[labels_test.item()])

cm = confusion_matrix(actual_arr, predict_arr, labels=class_names)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plot_util.plot_confusion_matrix(cm, labels=class_names, title="pest_41")

acc = accuracy_score(actual_arr, predict_arr)
pre = precision_score(actual_arr, predict_arr, average='weighted')
recall = recall_score(actual_arr, predict_arr, average='weighted')
f1score = f1_score(actual_arr, predict_arr, average='weighted')
print(f"accuracy={acc}, precision={pre}, recall={recall},f1score={f1score}")
