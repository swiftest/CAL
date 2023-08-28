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
    pred = pred.t()  # (1, batch) or (2, batch) or (3, batch), it is used to predict the 1st, 2nd, and 3rd most likely categories
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # (1, batch) or (2, batch) or (3, batch) 
    
    res = []  # The program here is very important. This res list is used to record the ratio of the total number of accurate predictions made in the previous few times (1, 2, or 3 times) to the total number of samples in the entire batch (i.e. accuracy)!!! Be sure to analyze it carefully!!!
    
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum()
        res.append(correct_k.mul_(100.0/batch_size))
    return res, target, pred.squeeze()


# test model
def test_epoch(model, test_loader):
    tar = np.array([]).astype('int')
    pre = np.array([]).astype('int')
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()
        
        batch_pred, _ = model(batch_data)  # (batch, 13, 16)
        batch_pred = safe_norm(batch_pred, axis=-1)  # (batch, 13)
        _, pred = batch_pred.topk(1, axis=1)  # (100, 1)
        pp = pred.squeeze()  # (100, )
        
        tar = np.append(tar, batch_target.data.cpu().numpy())
        pre = np.append(pre, pp.data.cpu().numpy())
    return tar, pre


def valid_epoch(model, true_loader):
    pre = np.array([]).astype('int')
    for batch_idx, (batch_data, batch_target) in enumerate(true_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()
        
        batch_pred, _ = model(batch_data)  # (100, 13, 16)
        batch_pred = safe_norm(batch_pred, axis=-1)  # (100, 13)
        _, pred = batch_pred.topk(1, axis=1)  # (100, 1)
        pp = pred.squeeze()  # (100, )
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre  # (100, )


# train model
def train_epoch(model, train_loader, criterion, optimizer):
    objs = AverageMeter()
    top1 = AverageMeter()
    tar = np.array([]).astype('int')
    pre = np.array([]).astype('int')
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   
        
        optimizer.zero_grad()
        batch_pred, batch_recon = model(batch_data, batch_target)  # (13, 13, 16)
        loss = criterion(batch_pred, batch_target, batch_data, batch_recon)
        loss.backward()
        optimizer.step()
        
        batch_pred = safe_norm(batch_pred, axis=-1)  # (13, 13)
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)  # Calculate the average loss of all training samples
        top1.update(prec1[0].data, n)  # Calculate the average accuracy of all training samples
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre


def plot_confusion_matrix(dataset, cm, path, cmap=mpl.cm.PuBu_r, normalize=True, dpi=300):
    font_style = dict(family='DejaVu Sans', weight='black', size=15)
    font_style_ticks = dict(family='DejaVu Sans', size=15)
    OA, AA_mean, Kappa, _ = cal_results(cm)
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure(figsize=(13, 10))
    ax = plt.subplot(111)
    plt.imshow(cm, cmap=cmap)
    plt.colorbar()
    if dataset == 'Pavia':
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Metal Sheets', 'Bare Soil', 'Bitumen', 'Bricks','Shadows']
    elif dataset == 'KSC':
        target_names = ["Scrub-1", "Willow-2", "Palm-3", "Pine-4", "Broadleaf-5", "Hardwood-6", "Swap-7", 
                        "Graminoid-8", "Spartina-9", "Cattail-10", "Salt-11", "Mud-12", "Water-13"]
    elif dataset == 'Chikusei':
        target_names = ['Water', 'School', 'Park', 'Farmland', 'Natural Plants', 'Weeds', 'Forest', 'Grass','Rice Field (grown)', 
                        'Rice Field (first stage)', 'Row crops', 'Plastic House', 'Manmade-1', 'Manmade-2', 'Manmade-3', 'Manmade-4', 
                        'Manmade Grass', 'Asphalt', 'Paved Ground']
    elif dataset == 'HU2013':
        target_names = ["Healthy Grass", "Stressed Grass", "Synthetic Grass", "Trees", "Soil", "Water", "Residential", "Commercial", "Road",
                        "Highway", "Railway", "Parkinig Lot 1", "Parkinig Lot 2", "Tennis Court", "Running Track"]
    else:
        raise ValueError("Unknown dataset")
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=90, **font_style_ticks)
    plt.yticks(tick_marks, target_names, **font_style_ticks)
    thresh = 0.001
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] > thresh:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]), horizontalalignment="center", color="black")

    #plt.ylabel('True label', **font_style)
    plt.xlabel('OA={:0.2f}%,  AA={:0.2f}%,  Kappa={:0.4f}'.format(OA*100., AA_mean*100., Kappa), **font_style)
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
    # Network outputs
    outputs = None
    def __init__(self, layer):
        # Register the callback function on the (layer_num) layer of the model and pass in the processing function.
        self.hook = layer.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, inputs, outputs):
        self.outputs = outputs.cpu()
    def remove(self):
        # Called by a callback handle, used to remove the callback function from the network layer.
        self.hook.remove()