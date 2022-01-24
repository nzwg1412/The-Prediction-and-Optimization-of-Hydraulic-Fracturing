import torch
import os
import time
import pandas as pd
import numpy as np
import copy
import random
from model.model import NetX2Y, NetH2Y

import torch.utils.data as Data
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from dataset.fracture_dataset2 import PDDataset2
from config.config import global_config
from sklearn.model_selection import train_test_split

plt.rcParams['font.size'] = 18
plt.rcParams['font.sans-serif'] = 'Times New Roman'


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 150 epochs"""
    lr = base_lr * (0.1 ** (epoch // 25))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# global config
data_path = global_config.getRaw('config', 'data_base_path')
stages = global_config.getRaw('config', 'stages')
runs_save_folder = os.path.join(global_config.getRaw('config', 'runs_save_folder'), stages)
model_save_folder = os.path.join(global_config.getRaw('config', 'model_save_folder'), stages)
trainer_name = global_config.getRaw('config', 'model_name')
best_h2y_model = global_config.getRaw('config', 'best_h2y_model')

if not os.path.exists(model_save_folder):
    os.makedirs(model_save_folder)

# train config
epochs = int(global_config.getRaw('train', 'num_epochs'))
batch_size = int(global_config.getRaw('train', 'batch_size'))
base_lr = float(global_config.getRaw('train', 'lr'))
save_freq = int(global_config.getRaw('train', 'save_freq'))
random_seed = int(global_config.getRaw('train', 'random_seed'))


# best_h2y_model = best_h2y_model.replace('h2y', 'h2y_noise_%.4f' % noise)

writer = SummaryWriter(os.path.join(runs_save_folder), '%s' % trainer_name)
model_dir = os.path.join(model_save_folder, '%s/' % trainer_name)

np.random.seed(random_seed)


def main_func_train_val(hidden1, hidden2, hidden3, hidden4, hidden_feat_len, out_feat_len, Hidden1, Hidden2, Hidden3, Hidden4,n_feature_len, noise, H_model=1, add_physical_info=1):
    # load data
    global ERROR
    file_path = os.path.join(data_path, '6_stages.csv')
    data = pd.read_csv(file_path)
    data.dropna(axis=0, how='any', inplace=True)
    if noise:
        data[['NPV']].apply(lambda x: (x + noise * np.std(x) * np.random.randn(x.shape[0])))
    data = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

    train_data, val_test_data = train_test_split(data, test_size=0.2, random_state=random_seed)
    test_data, val_data = train_test_split(val_test_data, test_size=0.5, random_state=random_seed)
    train_dataset = PDDataset2('6', train_data)
    val_dataset = PDDataset2('6', val_data)
    test_dataset = PDDataset2('6', test_data)

    train_loader = Data.DataLoader(
        dataset=train_dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,
        num_workers=0,
    )

    val_loader = Data.DataLoader(
        dataset=val_dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,
        num_workers=0,
    )


    net_h2y = NetH2Y(hidden1, hidden2, hidden3, hidden4, hidden_feat_len, out_feat_len)
    net_x2y = NetX2Y(Hidden1, Hidden2, Hidden3, Hidden4, hidden4, add_physical_info, n_feature_len, out_feat_len)


    loss_func = torch.nn.MSELoss()
    if H_model == 1:
        optimizer_h2y = torch.optim.Adam(net_h2y.parameters(), lr=base_lr)
        train_writer_h2y = SummaryWriter(os.path.join(runs_save_folder, trainer_name + '_h2y'))

        best_h2y_error = np.inf
        best_h2y_model = copy.deepcopy(net_h2y.state_dict())
        for epoch in range(epochs):
            adjust_learning_rate(optimizer_h2y, epoch)
            train(net_h2y, net_x2y, optimizer_h2y, loss_func, train_writer_h2y, train_loader, add_physical_info,
                  epoch, stage=1)
            h2y_error = val(net_h2y, net_x2y, loss_func, train_writer_h2y, val_loader, add_physical_info,
                            epoch, stage=1)
            if h2y_error < best_h2y_error:
                best_h2y_error = h2y_error
                best_h2y_model = copy.deepcopy(net_h2y.state_dict())
        ERROR = best_h2y_error
        torch.save(best_h2y_model, model_save_folder + "/%s_best_model_h2y.pth" % trainer_name)

    if H_model == 0:
        if add_physical_info:
            train_writer_x2y = SummaryWriter(
                os.path.join(runs_save_folder, trainer_name + '_x2y_added_noise_%.4f' % noise))
            save_x2y_model_path = model_save_folder + "/%s_best_model_x2y_added_noise_%.4f.pth" % (
                trainer_name, noise)
        else:
            train_writer_x2y = SummaryWriter(
                os.path.join(runs_save_folder, trainer_name + '_px2y_noise_%.4f' % noise))
            save_x2y_model_path = model_save_folder + "/%s_best_model_px2y_noise_%.4f.pth" % (trainer_name, noise)
        optimizer_x2y = torch.optim.Adam(net_x2y.parameters(), lr=base_lr)

        best_x2y_error = np.inf
        best_x2y_model = copy.deepcopy(net_x2y.state_dict())
        for epoch in range(epochs):
            adjust_learning_rate(optimizer_x2y, epoch)
            train(net_h2y, net_x2y, optimizer_x2y, loss_func, train_writer_x2y, train_loader, add_physical_info,
                  epoch, stage=2)
            x2y_error = val(net_h2y, net_x2y, loss_func, train_writer_x2y, val_loader, add_physical_info,
                            epoch, stage=2)
            if x2y_error < best_x2y_error:
                best_x2y_error = x2y_error
                best_x2y_model = copy.deepcopy(net_x2y.state_dict())
        torch.save(best_x2y_model, save_x2y_model_path)
        ERROR = best_x2y_error
    return ERROR


def train(model_1, model_2, optimizer, loss_func, train_writer, loader, add_physical_info, epoch, stage):
    model_1.train()
    model_2.train()

    losses = []
    pred, target = [], []
    epoch_start_time = time.time()
    for step, (x, h, y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
        start_time = time.time()
        if stage == 1:
            prediction = model_1(h)
            loss = loss_func(prediction, y)
        elif stage == 2:
            if add_physical_info:
                model_1.load_state_dict(torch.load(model_save_folder + "/%s" % best_h2y_model))
                model_1.eval()
                _, physical_info = model_1(h)
                model_2.add_physical_info(physical_info)
                prediction = model_2(x)
            else:
                prediction = model_2(x)
            loss = loss_func(prediction, y)
        else:
            print("please input the correct stage")
            return
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.item())
        if step == 0:
            pred = prediction.detach().numpy()
            target = y.detach().numpy()
        else:
            pred = np.concatenate((np.array(pred), prediction.detach().numpy()), 0)
            target = np.concatenate((np.array(target), y.detach().numpy()), 0)
        print(
            f"Stage: {stage}\t Epoch: {epoch} \t Batch_num: {step} \t Loss={loss.data.cpu():.4} \t "
            f"Time={time.time() - start_time:.4}")
    error = np.linalg.norm(target - pred, 2) / np.linalg.norm(target, 2)
    print(f"Train \t Epoch={epoch} \t AVG_Loss={np.mean(losses):.4} \t Time={time.time() - epoch_start_time:.4} \t"
          f"l2_error={error:.4}")
    train_writer.add_scalar('Loss/train', np.mean(losses), epoch)
    train_writer.add_scalar('l2_error/train', error, epoch)
    train_writer.flush()
    # return np.mean(losses)


def val(model_1, model_2, loss_func, train_writer, loader, add_physical_info, epoch, stage):
    model_1.eval()
    model_2.eval()
    with torch.no_grad():
        losses = []
        pred, target = [], []
        epoch_start_time = time.time()
        for step, (x, h, y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
            start_time = time.time()
            if stage == 1:
                prediction, _ = model_1(h)
                loss = loss_func(prediction, y)
            elif stage == 2:
                if add_physical_info:
                    model_1.load_state_dict(torch.load(model_save_folder + "/%s" % best_h2y_model))

                    _, physical_info = model_1(h)
                    model_2.add_physical_info(physical_info)
                    prediction = model_2(x)
                else:
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
            print(
                f"Stage: {stage}\t Epoch: {epoch} \t Batch_num: {step} \t Loss={loss.data.cpu():.4} \t "
                f"Time={time.time() - start_time:.4}")

        error = np.linalg.norm(target - pred, 2) / np.linalg.norm(target, 2)
        # if epoch == epochs - 1:
        #     lilizhetomakesurethedifference = pd.DataFrame(np.concatenate((pred, target), 1))
        #     lilizhetomakesurethedifference.to_csv("lilizhetomakesurethedifference" + str(epoch) + ".csv")
        print(f"Val \t Epoch={epoch} \t AVG_Loss={np.mean(losses):.4} \t Time={time.time() - epoch_start_time:.4} \t"
              f" l2_error={error:.4}")
        train_writer.add_scalar('Loss/val', np.mean(losses), epoch)
        train_writer.add_scalar('l2_error/val', error, epoch)
        train_writer.flush()
    return error

# if __name__ == '__main__':
#     main_func_train_val(20, 40, 20, 20)
