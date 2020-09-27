#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_weighted_cross.py
@Time    :   2020/09/18 13:41:48
@Author  :   shengshijieshao
@Version :   1.0
@Contact :   2422656558@qq.com
'''

# backbone 改为 resnet50，加载了预训练模型，且冻结了resnet的参数
# 其余设置同train_FocalLoss.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from model.fcn import VGGNet, FCNs
from dataset.lane_cls_data import LaneClsDataset
from metrics.evaluator import Evaluator
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
from Focal_Loss import BCEFocalLoss
from model.resnet import resnet50
from model.resnet50_fcn import FCNs


BATCH_SIZE = 4
LR = 1e-3
MAX_EPOCH = 50
IMG_H = 288
IMG_W = 800
SAVE_INTERVAL = 5
MODEL_CKPT_DIR = "./ckpt/resnet_fcn/"
FIGURE_DIR = "./figures/resnet_fcn/"
WARMUP_STEPS = 88
WARMUP_FACTOR = 1.0 / 3.0
lr_schedule = [264, 880]

def lr_func(step, lr):   
    if step < WARMUP_STEPS:
        alpha = float(step) / WARMUP_STEPS
        warmup_factor = WARMUP_FACTOR * (1.0 - alpha) + alpha
        lr = lr*warmup_factor
    else:
        for i in range(len(lr_schedule)):
            if step < lr_schedule[i]:
                break
            lr *= 0.1
    return float(lr)

def draw_figure(record_dict, title="Loss", ylabel='loss', filename="loss.png"):
    plt.clf()
    epochs = np.arange(0, MAX_EPOCH)
    plt.plot(epochs, record_dict['train'], color='red', linewidth=1, label='train')
    plt.plot(epochs, record_dict['val'], color='blue', linewidth=1, label='val')
    plt.xticks(np.arange(0, MAX_EPOCH, 5))
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='best')
    save_path = os.path.join(FIGURE_DIR, filename)
    plt.savefig(save_path)
    print('曲线图成功保存至{}'.format(save_path))


def main():
    # get model，第一步是判断设备cpu还是GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 获取backbone
    fcn_model = FCNs(pretrained_net=resnet50(pretrained=True))
    # criterion = nn.BCELoss()
    criterion = BCEFocalLoss()
    # 优化函数
    optimizer = optim.Adam(fcn_model.parameters(), lr=LR, weight_decay=0.0001)
    # 评估器
    evaluator = Evaluator(num_class=2)

    if device == 'cuda':
        fcn_model.to(device)
        criterion.to(device)

    # get dataloader
    train_set = LaneClsDataset(list_path='train.tsv',
                               dir_path='data_road',
                               img_shape=(IMG_W, IMG_H))
    # DataLoader是pytorch里面内置的处理类，对于数据集以及训练属性进行封装，
    # 训练属性是指batch_size（一次扔多少个样本进模型进行train），shuffle（数据是否打乱）；
    # num_worker则是指多少个进程（subprocesses，我觉得线程更加合适一些）
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=8)

    val_set = LaneClsDataset(list_path='val.tsv',
                             dir_path='data_road',
                             img_shape=(IMG_W, IMG_H))
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)

    # info records
    # 指定了一堆字段，后面用于赋值，
    # TODO 看完了再来补充
    loss_dict = defaultdict(list)
    px_acc_dict = defaultdict(list)
    mean_px_acc_dict = defaultdict(list)
    mean_iou_dict = defaultdict(list)
    freq_iou_dict = defaultdict(list)

    for epoch_idx in range(1, MAX_EPOCH + 1):
        # train stage
        # 这里设置当前模式为“train”，对应是“evaluation”，这个设置对于某些模块是影响很大的
        # 比如dropout，batch_normalization
        fcn_model.train()
        evaluator.reset()
        train_loss = 0.0
        # 注意这里体现了Loader的优势，因为指定了batchsize，所以loader只会返回batch_size大小的数据
        # 这里观察依稀返回每个item的组成，分别是id，一个二元组，图像和标签（label），现在知道为什么
        # 吃内存了，如果内存足够大，batch_size设置的足够大，那么每次就可以训练多个；但是小的batch_size
        for batch_idx, (image, label) in enumerate(train_loader):

            lr = LR
            lr = lr_func((epoch_idx-1) * 88 + batch_idx, lr)
            for param in optimizer.param_groups:
                param['lr']=lr

            image = image.to(device)
            # print(label.shape)
            # label = label.reshape(BATCH_SIZE, 288, 800)
            label = label.to(device)
            optimizer.zero_grad()
            output = fcn_model(image)
            output = torch.sigmoid(output)

            loss = criterion(output, label.long())
            loss.backward()

            evaluator.add_batch(torch.argmax(output, dim=1).cpu().numpy(),
                                torch.argmax(label, dim=1).cpu().numpy())
            train_loss += loss.item()
            print("[Train][Epoch] {}/{}, [Batch] {}/{}, [lr] {:.6f},[Loss] {:.6f}".format(epoch_idx,
                                                                              MAX_EPOCH,
                                                                              batch_idx+1,
                                                                              len(train_loader),
                                                                              lr,
                                                                              loss.item()))
            optimizer.step()

        ## 下面的dict的复制都是针对于评估参数的，用于记录评估模型进展（通过dict进行记录中间过程）
        # 以及出图
        loss_dict['train'].append(train_loss/len(train_loader))
        # TODO 像素精度（不是很懂原理，后面研究一下），为什么会是多个值呢？基于混淆矩阵的
        px_acc = evaluator.Pixel_Accuracy() * 100
        px_acc_dict['train'].append(px_acc)
        # TODO 像素精度均值（不是很懂，后面研究一下）
        # 是将px_acc的值取了均值，这说明上面获得px_acc其实是一个数组（多个值）
        mean_px_acc = evaluator.Pixel_Accuracy_Class() * 100
        mean_px_acc_dict['train'].append(mean_px_acc)
        mean_iou = evaluator.Mean_Intersection_over_Union() * 100
        mean_iou_dict['train'].append(mean_iou)
        freq_iou = evaluator.Frequency_Weighted_Intersection_over_Union() * 100
        freq_iou_dict['train'].append(freq_iou)
        print("[Train][Epoch] {}/{}, [PA] {:.2f}%, [MeanPA] {:.2f}%, [MeanIOU] {:.2f}%, ""[FreqIOU] {:.2f}%".format(
            epoch_idx,
            MAX_EPOCH,
            px_acc,
            mean_px_acc,
            mean_iou,
            freq_iou))
        # 再开始新的一轮epoch之前重置evaluator，就是将混淆矩阵设置为全0
        # TODO 为什么混淆矩阵初始化要*2，即为什么是2维的呢？
        evaluator.reset()
        # validate stage
        # eval是evaluation的简写，其效果和model.train(false)等价，这个属性针对dropout，batchnorm等
        # 层有效，进入推断模式这些层不需要参与计算，即这些层只是用于在训练阶段提供更大随机性和健壮性，
        # 在模型生成阶段已提高模型的性能；在预测（推断）阶段不需要他们
        fcn_model.eval()
        # 每一个epoch都是需要跑一下validation数据，验证本轮epoch的效果怎么样，因为进入到了valiation
        # 阶段，即预测（推断）阶段，所以上面设置为eval模式
        with torch.no_grad():
            val_loss = 0.0
            # 注意，这里image是batch_size个image（此次代码设置为1），是一个batch的validation数据集
            for image, label in val_loader:
                image, label = image.to(device), label.to(device)
                output = fcn_model(image)
                output = torch.sigmoid(output)
                loss = criterion(output, label.long())
                val_loss += loss.item()
                evaluator.add_batch(torch.argmax(output, dim=1).cpu().numpy(),
                                    torch.argmax(label, dim=1).cpu().numpy())
            # 求验证集loss均值
            val_loss /= len(val_loader)
            loss_dict['val'].append(val_loss)
            px_acc = evaluator.Pixel_Accuracy() * 100
            px_acc_dict['val'].append(px_acc)
            mean_px_acc = evaluator.Pixel_Accuracy_Class() * 100
            mean_px_acc_dict['val'].append(mean_px_acc)
            mean_iou = evaluator.Mean_Intersection_over_Union() * 100
            mean_iou_dict['val'].append(mean_iou)
            freq_iou = evaluator.Frequency_Weighted_Intersection_over_Union() * 100
            freq_iou_dict['val'].append(freq_iou)
            print("[Val][Epoch] {}/{}, [Loss] {:.6f}, [PA] {:.2f}%, [MeanPA] {:.2f}%, "
                  "[MeanIOU] {:.2f}%, ""[FreqIOU] {:.2f}%".format(epoch_idx,
                                                                  MAX_EPOCH,
                                                                  val_loss,
                                                                  px_acc,
                                                                  mean_px_acc,
                                                                  mean_iou,
                                                                  freq_iou))

        # save model checkpoints
        # 每隔SAVE_INTERVAL个epoch，就将checkpoint进行保存；或者达到了epoch的最大次数（即将退出训练）
        if epoch_idx % SAVE_INTERVAL == 0 or epoch_idx == MAX_EPOCH:
            os.makedirs(MODEL_CKPT_DIR, exist_ok=True)
            ckpt_save_path = os.path.join(MODEL_CKPT_DIR, 'epoch_{}.pth'.format(epoch_idx))
            torch.save(fcn_model.state_dict(), ckpt_save_path)
            print("[Epoch] {}/{}, 模型权重保存至{}".format(epoch_idx, MAX_EPOCH, ckpt_save_path))

    # draw figures
    os.makedirs(FIGURE_DIR, exist_ok=True)
    draw_figure(loss_dict, title='Loss', ylabel='loss', filename='loss.png')
    draw_figure(px_acc_dict, title='Pixel Accuracy', ylabel='pa', filename='pixel_accuracy.png')
    draw_figure(mean_px_acc_dict, title='Mean Pixel Accuracy', ylabel='mean_pa', filename='mean_pixel_accuracy.png')
    draw_figure(mean_iou_dict, title='Mean IoU', ylabel='mean_iou', filename='mean_iou.png')
    draw_figure(freq_iou_dict, title='Freq Weighted IoU', ylabel='freq_weighted_iou', filename='freq_weighted_iou.png')


if __name__ == "__main__":
    main()
