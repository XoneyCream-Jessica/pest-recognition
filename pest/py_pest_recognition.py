import datetime
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import transforms

import common.file as common_file
import common.logger as logger
import common.plot_util as plot_util
import common.torch_util as torch_util
from PIL import Image

logging = logger.get_logger(__name__)

data_dir = os.path.join(common_file.get_data_path(), "41all")
train_dir = os.path.join(data_dir, 'train')
im_height = 224
im_width = 224
batch_size = 32
epochs = 60
class_num = 41
device = torch_util.get_device()
# 优化点1 HyperParam
lr = 1e-5
weight_decay = 3e-4
drop_out = 0.6
early_stop = 10
logging.info(
    f"[ HyperParam ]| device={device} | batch_size={batch_size} | total_epochs={epochs} | learning_rate={lr} | "
    f"weight_decay={weight_decay} | "
    f"drop_out={drop_out} | early_stop={early_stop}")

# 优化点2 数据增强
train_tf = transforms.Compose(
    [
        transforms.Resize((im_height, im_width)),
        transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
        ]),
        transforms.RandomApply(
            [transforms.RandomChoice([
                transforms.RandomRotation(45),
                transforms.RandomRotation(315),
                transforms.RandomAffine(0, shear=20),
                transforms.RandomAffine(degrees=0, scale=(0.7, 0.7)),
                transforms.RandomAffine(degrees=0, scale=(1.3, 1.3)),
                transforms.RandomAffine(degrees=0, translate=(0.3, 0)),
                transforms.RandomAffine(degrees=0, translate=(0, 0.3)),
                transforms.RandomAffine(degrees=0, translate=(0.3, 0.3)),
            ])], p=0.95),
        transforms.ToTensor()
    ]
)

# test transforms
# test_img = Image.open(os.path.join(os.path.join(train_dir, 'limax'), 'limax0.jpg'))
#
# test_tf = transforms.Compose(
#     [
#         transforms.RandomAffine(degrees=0, shear=0.2),
#         transforms.ToTensor(),
#         transforms.ToPILImage()
#     ]
# )
# test_tf1 = transforms.Compose(
#     [
#         transforms.RandomRotation(45),
#         transforms.ToTensor(),
#         transforms.ToPILImage()
#     ]
# )
# test_tf2 = transforms.Compose(
#     [
#         transforms.RandomRotation(315),
#         transforms.ToTensor(),
#         transforms.ToPILImage()
#     ]
# )
# test_tf_img = test_tf(test_img)
# test_tf_img.show()
# test_tf_img = test_tf1(test_img)
# test_tf_img.show()
# test_tf_img = test_tf2(test_img)
# test_tf_img.show()
# image augmentation
train_dataset = torchvision.datasets.ImageFolder(root=train_dir,
                                                 transform=train_tf)

val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'val'),
                                               transform=transforms.Compose(
                                                   [
                                                       transforms.Resize((im_height, im_width)),
                                                       transforms.ToTensor()
                                                   ]))
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

class_names = train_dataset.classes
logging.info('class_names:{}'.format(class_names))

model = models.resnet50(pretrained=True)

# 优化点3 冻结层的调整
for name, param in model.named_parameters():
    # 层数调整
    # if ("layer3" in name) or ("layer4" in name) or ("fc" in name):
    # 细分隐藏层的调整
    # if ("layer4.0" in name) or ("layer4.1" in name) or ("layer4.2" in name) or ("fc" in name):
    if ("layer2" in name) or ("layer3" in name) or ("layer4" in name) or ("fc" in name):
        param.requires_grad = True
    else:
        param.requires_grad = False
# 全连接层的输入通道in_channels个数
num_fc_in = model.fc.in_features

# 改变全连接层，41分类问题，out_features = 41,增加一层drop_out
model.fc = nn.Sequential(nn.Dropout(drop_out),
                         nn.Linear(num_fc_in, class_num))

model = model.to(device)

# 定义损失函数
loss_fc = nn.CrossEntropyLoss()

# 选择优化方法
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

train_loss_record = []
train_acc_record = []
valid_loss_record = []
valid_acc_record = []
last_save_model_loss = 10000
last_save_model_acc = 0
no_update_loss_count = 0

start_time = time.time()
for epoch in range(epochs):
    model.train()
    train_losses = []
    train_accs = []
    for images_train, images_label in train_dataloader:
        # GPU/CPU
        inputs = images_train.to(device)
        labels = images_label.to(device)
        optimizer.zero_grad()
        # forward
        outputs = model(inputs)
        # loss
        loss = loss_fc(outputs, labels)
        # loss求导，反向
        loss.backward()
        # 优化
        optimizer.step()
        # Compute the accuracy for current batch.
        acc = (outputs.argmax(dim=-1) == labels).float().mean().item()

        # Record the loss and accuracy.
        train_losses.append(loss.item())
        train_accs.append(acc)

        # The average loss and accuracy of the training set is the average of the recorded values.
    train_loss = sum(train_losses) / len(train_losses)
    train_acc = sum(train_accs) / len(train_accs)
    # Print the information
    train_loss_record.append(train_loss)
    train_acc_record.append(train_acc)
    logging.info(f"[ Train | {epoch + 1:03d}/{epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    model.eval()

    valid_losses = []
    valid_accs = []

    for images_test, labels_test in val_dataloader:
        images_test = images_test.to(device)
        labels_test = labels_test.to(device)
        with torch.no_grad():
            outputs_test = model(images_test)
        loss = loss_fc(outputs_test, labels_test)

        # Compute the accuracy for current batch.
        acc = (outputs_test.argmax(dim=-1) == labels_test).float().mean().item()

        # Record the loss and accuracy.
        valid_losses.append(loss.item())
        valid_accs.append(acc)
    valid_loss = sum(valid_losses) / len(valid_losses)
    valid_acc = sum(valid_accs) / len(valid_accs)
    valid_loss_record.append(valid_loss)
    valid_acc_record.append(valid_acc)
    # Print the information.
    logging.info(f"[ Valid | {epoch + 1:03d}/{epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    if valid_loss < last_save_model_loss:
        torch.save(model.state_dict(), os.path.join(common_file.get_models_path(), 'pest_41.pth'))
        logging.info(f"Update model when {valid_loss:.5f} < {last_save_model_loss:.5f}")
        last_save_model_loss = valid_loss
        last_save_model_acc = valid_acc
        no_update_loss_count = 0
    else:
        no_update_loss_count = no_update_loss_count + 1
        if no_update_loss_count > early_stop:
            logging.info("Early stop !")
            break

loss_record = {'train': train_loss_record, 'val': valid_loss_record}
acc_record = {'train': train_acc_record, 'val': valid_acc_record}
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
title = "pest41." + timestamp
plot_util.plot_learning_curve(loss_record, title=title)
plot_util.plot_acc_curve(acc_record, title=title)

logging.info(f"Training finish ! cost={time.time() - start_time},last_save_model_loss={last_save_model_loss},last_save_model_acc={last_save_model_acc}")
