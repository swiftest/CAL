import torch
import itertools
import numpy as np
import torch.nn as nn
import matplotlib as mpl
from matplotlib import colors
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def safe_norm(s, axis=-1, epsilon=1e-7, keep_dim=False):
    squared_norm = torch.sum(torch.square(s), axis=axis, keepdim=keep_dim)
    return torch.sqrt(squared_norm + epsilon)


def losspredloss(input, target, margin=1.0, reduction='mean'):
    assert input.size(0) == target.size(0)
    assert input.device == target.device
    device = input.device
    batch_size = input.size(0)
    if batch_size % 2 != 0:
        input_expand = torch.zeros(batch_size+1).to(device)
        input_expand[:batch_size//2+1] = input[:batch_size//2+1]
        input_expand[batch_size//2 + 2:] = input[batch_size//2+1:]
        input_expand[batch_size//2+1] = input[0]
        input = input_expand
        
        target_expand = torch.zeros(batch_size+1).to(device)
        target_expand[:batch_size//2+1] = target[:batch_size//2+1]
        target_expand[batch_size//2 + 2:] = target[batch_size//2+1:]
        target_expand[batch_size//2+1] = target[0]
        target = target_expand
    input = (input - input.flip(0))[:len(input)//2]
    # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()
    
    ones = 2 * torch.sign(torch.clamp(target, min=0)) - 1
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - ones * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    return loss


class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    # output: (batch, 13)
    # target: (batch, )
    maxk = max(topk)  # 1
    batch_size = target.size(0)  # batch
    
    _, pred = output.topk(maxk, axis=1)  # (batch, 1) or (batch, 2) or (batch, 3)
    pred = pred.t()  # (1, batch) or (2, batch) or (3, batch) 用来预测1st, 2nd, 3th高的可能的类别
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # (1, batch) or (2, batch) or (3, batch) 
    
    res = []  # 这里的程序非常重要，这个res列表用来记录前几次（1次2次或3次）预测准确的总个数与整个批次总样本数的比值（也就是准确率）！！！一定要好好分                 析！！！
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum()
        res.append(correct_k.mul_(100.0/batch_size))
    return res, target, pred.squeeze()


def test_epoch(models, test_loader):
    tar = np.array([]).astype('int')
    pre = np.array([]).astype('int')
    with torch.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
            batch_data = batch_data.cuda()
            batch_target = batch_target.cuda()
        
            batch_pred, _, _ = models['backbone'](batch_data)  # (batch, 13, 16)
            batch_pred = safe_norm(batch_pred, axis=-1)  # (batch, 13)
            _, pred = batch_pred.topk(1, axis=1)  # (100, 1)
            pp = pred.squeeze()  # (100, )
        
            tar = np.append(tar, batch_target.data.cpu().numpy())
            pre = np.append(pre, pp.data.cpu().numpy())
    return tar, pre


def valid_epoch(models, true_loader):
    pre = np.array([]).astype('int')
    with torch.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(true_loader):
            batch_data = batch_data.cuda()
            batch_target = batch_target.cuda()
        
            batch_pred, _, _ = models['backbone'](batch_data)  # (100, 13, 16)
            batch_pred = safe_norm(batch_pred, axis=-1)  # (100, 13)
            _, pred = batch_pred.topk(1, axis=1)  # (100, 1)
            pp = pred.squeeze()  # (100, )
            pre = np.append(pre, pp.data.cpu().numpy())
    return pre  # (100, )


def train_epoch(models, train_loader, criterion, optimizers):
    objs_target = AverageMeter()
    top1 = AverageMeter()
    tar = np.array([]).astype('int')
    pre = np.array([]).astype('int')
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()
        
        batch_pred, batch_recon, features = models['backbone'](batch_data, batch_target)  # (batch_size, 13, 16)
        target_loss = criterion(batch_pred, batch_target, batch_data, batch_recon)  # (batch_size, )
        
        features[0] = features[0].detach()
        features[1] = features[1].detach()
        features[2] = features[2].detach()
        features[3] = features[3].detach()
        
        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))
        
        m_backbone_loss = torch.mean(target_loss)
        m_module_loss   = losspredloss(pred_loss, target_loss)
        
        m_backbone_loss.backward()
        m_module_loss.backward()
        
        optimizers['backbone'].step()
        optimizers['module'].step()
        
        batch_pred = safe_norm(batch_pred, axis=-1)  # (13, 13)
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs_target.update(m_backbone_loss.data, n)  # 计算所有训练样本的平均损失
        top1.update(prec1[0].data, n)  # 计算所有训练样本的平均准确率
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs_target.avg, tar, pre


def plot_confusion_matrix(dataset, cm, path, cmap=mpl.cm.PuBu_r, normalize=True, dpi=300):
    font_style = dict(family='Times New Roman', weight='black', size=20)
    font_style_ticks = dict(family='Times New Roman', size=15)
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure(figsize=(13, 10))
    ax = plt.subplot(111)
    plt.imshow(cm, cmap=cmap)
    plt.colorbar()
    if dataset == 'KSC':
        target_names = ["Scrub-1", "Willow-2", "Palm-3", "Pine-4", "Broadleaf-5", "Hardwood-6", "Swap-7", 
                        "Graminoid-8", "Spartina-9", "Cattail-10", "Salt-11", "Mud-12", "Water-13"]
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=90, **font_style_ticks)
    plt.yticks(tick_marks, target_names, **font_style_ticks)
    thresh = 0.001
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] > thresh:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]), horizontalalignment="center", color="black")

    plt.ylabel('True label', **font_style)
    plt.xlabel('Predicted label \n accuracy(OA)={:0.2f}%, misclass={:0.2f}%'.format(accuracy*100., misclass*100.), **font_style)
    plt.show()
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    
    
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


def cal_results(matrix):
    shape = np.shape(matrix)  # 13 * 13
    number = 0
    total_sum = 0
    AA = np.zeros([shape[0]], dtype=float)  # (13, )
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])  # recall ratio
        total_sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = total_sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 2:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 3:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 4:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 5:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 6:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 7:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 8:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 9:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 10:
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 11:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 12:
            y[index] = np.array([128, 0, 128]) / 255.
        if item == 13:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 14:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 15:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 16:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 17:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 18:
            y[index] = np.array([215, 255, 0]) / 255.
        if item == 19:
            y[index] = np.array([0, 255, 215]) / 255.
    return y


def classification_map(pre_map, save_path, dpi=300):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(pre_map.shape[1] * 3.0 / dpi, pre_map.shape[0] * 3.0 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(pre_map)
    fig.savefig(save_path, dpi=dpi)
    
    
def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k,v))
        
        
class ActivationOutputData():
    # 网络输出值
    outputs = None
    def __init__(self, layer):
        # 在模型的layer_num层上注册回调函数，并传入处理函数
        self.hook = layer.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, inputs, outputs):
        self.outputs = outputs.cpu()
    def remove(self):
        # 由回调句柄调用，用于将回调函数从网络层删除
        self.hook.remove()
