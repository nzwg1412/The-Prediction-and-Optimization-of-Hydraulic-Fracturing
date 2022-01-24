"""
Date: 2021/05/10
Author: worith
"""

import torch
import argparse
import os
import time
import numpy as np
import pandas as pd
import copy
import random
from model.model import NetX2Y, NetH2Y

import torch.utils.data as Data
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from dataset.fracture_dataset import PDDataset
from config.config import global_config
from sklearn.model_selection import train_test_split

plt.rcParams['font.size'] = 18
plt.rcParams['font.sans-serif'] = 'Times New Roman'


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 150 epochs"""
    lr = base_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# global config
data_path = global_config.getRaw('config', 'data_base_path')
stages = global_config.getRaw('config', 'stages')
runs_save_folder = os.path.join(global_config.getRaw('config', 'runs_save_folder'), stages)
model_save_folder = os.path.join(global_config.getRaw('config', 'model_save_folder'), stages)
best_h2y_model = global_config.getRaw('config', 'best_h2y_model')
best_x2y_added_model = global_config.getRaw('config', 'best_x2y_added_model')
best_x2y_model = global_config.getRaw('config', 'best_x2y_model')
# H_model = int(global_config.getRaw('config', 'is_model_h2y'))
trainer_name = global_config.getRaw('config', 'model_name')

if not os.path.exists(model_save_folder):
    os.makedirs(model_save_folder)

# train config
epochs = int(global_config.getRaw('train', 'num_epochs'))
batch_size = int(global_config.getRaw('train', 'batch_size'))
base_lr = float(global_config.getRaw('train', 'lr'))
save_freq = int(global_config.getRaw('train', 'save_freq'))
random_seed = int(global_config.getRaw('train', 'random_seed'))


# writer = SummaryWriter(os.path.join(runs_save_folder), '%s' % args.trainer_name)
model_dir = os.path.join(model_save_folder, '%s/' % trainer_name)


def main_func_test(hidden1, hidden2, hidden3, hidden4, hidden_feat_len, out_feat_len, Hidden1, Hidden2, Hidden3, Hidden4, n_feature_len, noise, H_model=1, add_physical_info=1):
    # load data
    global ERROR
    file_path = os.path.join(data_path, '6_stages.csv')
    data = pd.read_csv(file_path)
    data.dropna(axis=0, how='any', inplace=True)
    if noise:
        data[['NPV']].apply(lambda x: (x + noise * np.std(x) * np.random.randn(x.shape[0])))
    data = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    # if noise:
    #     noise_data = np.array(data['Fracture Spacing'])
    #     noise_data = noise_data + noise * \
    #                  np.std(noise_data) * np.random.randn(noise_data.shape[0])
    #     data['Fracture Spacing'] = noise_data
    train_data, val_test_data = train_test_split(data, test_size=0.2, random_state=random_seed)
    test_data, val_data = train_test_split(val_test_data, test_size=0.5, random_state=random_seed)
    test_dataset = PDDataset('6', test_data)
    test_loader = Data.DataLoader(
        dataset=test_dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,
        num_workers=0,
    )

    net_h2y = NetH2Y(hidden1, hidden2, hidden3, hidden4, hidden_feat_len, out_feat_len)
    net_x2y = NetX2Y(Hidden1, Hidden2, Hidden3, Hidden4, hidden1, add_physical_info, n_feature_len, out_feat_len)

    loss_func = torch.nn.MSELoss()
    if H_model == 1:
        ERROR = test(net_h2y, net_x2y, loss_func, test_loader, add_physical_info, noise,
             stage=1)

    if H_model == 0:
        ERROR = test(net_h2y, net_x2y, loss_func, test_loader, add_physical_info, noise,
                            stage=2)
    return ERROR


def test(model_1, model_2, loss_func, loader, add_physical_info, noise, stage):
    model_1.eval()
    model_2.eval()
    with torch.no_grad():
        losses = []
        pred, target = [], []
        epoch_start_time = time.time()
        text_name = ""
        for step, (x, h, y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
            if stage == 1:
                text_name = "h2y"
                model_1.load_state_dict(torch.load(model_save_folder + "/%s" % best_h2y_model))
                prediction, _ = model_1(h)
                loss = loss_func(prediction, y)
            elif stage == 2:
                if add_physical_info:
                    text_name = "x2y_added"
                    model_1.load_state_dict(torch.load(model_save_folder + "/%s" % best_h2y_model))
                    _, physical_info = model_1(h)
                    model_2.load_state_dict(torch.load(
                        model_save_folder + "/%s_best_model_x2y_added_noise_%.4f.pth" % (trainer_name, noise)))
                    model_2.add_physical_info(physical_info)
                    prediction = model_2(x)
                else:
                    text_name = "x2y"
                    model_2.load_state_dict(torch.load(
                        model_save_folder + "/%s_best_model_x2y_noise_%.4f.pth" % (trainer_name, noise)))
                    prediction = model_2(x)
                loss = loss_func(prediction, y)
            else:
                print("please input the correct stage")
                return
            losses.append(loss.data.item())
            if step == 0:
                pred = prediction.detach().numpy()
                target = y.detach().numpy()
            else:
                pred = np.concatenate((np.array(pred), prediction.detach().numpy()), 0)
                target = np.concatenate((np.array(target), y.detach().numpy()), 0)
            # print(
            #     f"Stage: {stage}\t Epoch: {epoch} \t Batch_num: {step} \t Loss={loss.data.cpu():.4} \t "
            #     f"Time={time.time() - start_time:.4}")
        error = np.linalg.norm(target - pred, 2) / np.linalg.norm(target, 2)
        print(f"Test of {text_name}: AVG_Loss={np.mean(losses):.4} \t Time={time.time() - epoch_start_time:.4} \t"
              f" l2_error={error:.4}")
    return error

#
# if __name__ == '__main__':
#     main()
