import numpy as np


def choose_train_and_test_proportion(groundtruth, proportion_per_class, seed):
    rs = np.random.RandomState(seed)
    
    num_classes = np.max(groundtruth)
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    
    for i in range(num_classes):
        each_class = np.argwhere(groundtruth == i+1)  # 返回标签i的位置索引数组
        quantity = np.ceil(each_class.shape[0] * proportion_per_class).astype(np.int32)
        rs.shuffle(each_class)
        pos_train[i] = each_class[:quantity]
        number_train.append(quantity)
        pos_test[i] = each_class[quantity:]
        number_test.append(pos_test[i].shape[0])  # 每一类测试样本的个数
        pos_true[i] = each_class
        number_true.append(each_class.shape[0])
    
    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]]
    total_pos_train = total_pos_train.astype(int)
    
    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]]
    total_pos_test = total_pos_test.astype(int)
    
    total_pos_true = pos_true[0]
    for i in range(1, num_classes):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)
    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true


def choose_train_and_test_number(groundtruth, num_train_per_class, seed):  # divide dataset into train and test datasets
    rs = np.random.RandomState(seed)
    
    num_classes = np.max(groundtruth)
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    
    for i in range(num_classes):
        each_class = np.argwhere(groundtruth == i+1)  # 返回标签i的位置索引数组
        rs.shuffle(each_class)
        pos_train[i] = each_class[:num_train_per_class]
        number_train.append(num_train_per_class)  # 每一类训练样本的个数
        pos_test[i] = each_class[num_train_per_class:]
        number_test.append(pos_test[i].shape[0])  # 每一类测试样本的个数
        pos_true[i] = each_class
        number_true.append(each_class.shape[0])
    
    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]]
    total_pos_train = total_pos_train.astype(int)
    
    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]]
    total_pos_test = total_pos_test.astype(int)
    
    total_pos_true = pos_true[0]
    for i in range(1, num_classes):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)
    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true


# 边界拓展：镜像
def mirror_hsi(height, width, band, data, patch_size=9):
    padding = patch_size // 2
    mirror_hsi = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)
    # 中心区域
    mirror_hsi[padding:(padding+height), padding:(padding+width), :] = data
    # 左边镜像
    for i in range(padding):
        mirror_hsi[padding:(padding+height), i, :] = data[:, padding-i-1, :]
    # 右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding), width+padding+i, :] = data[:, width-1-i, :]
    # 上边镜像
    for i in range(padding):
        mirror_hsi[i, :, :] = mirror_hsi[padding*2-i-1, :, :]
    # 下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i, :, :] = mirror_hsi[height+padding-1-i, :, :]
    print("**************************************************")
    print("patch_size is : {}".format(patch_size))
    print("mirror_data shape : [{0}, {1}, {2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi


# 获取patch的图像数据
def gain_neighborhood_pixel(mirror_data, pos, i, patch_size):
    x = pos[i, 0]
    y = pos[i, 1]
    temp_image = mirror_data[x:(x+patch_size), y:(y+patch_size), :]
    return temp_image


# 汇总训练数据和测试数据
def train_and_test_data(mirror_data, band, train_pos, test_pos, true_pos, patch_size=9):
    x_train = np.zeros((train_pos.shape[0], patch_size, patch_size, band), dtype=float)  # (695, 9, 9, 176)
    x_test = np.zeros((test_pos.shape[0], patch_size, patch_size, band), dtype=float)
    x_true = np.zeros((true_pos.shape[0], patch_size, patch_size, band), dtype=float)
    for i in range(train_pos.shape[0]):
        x_train[i] = gain_neighborhood_pixel(mirror_data, train_pos, i, patch_size)
    for j in range(test_pos.shape[0]):
        x_test[j] = gain_neighborhood_pixel(mirror_data, test_pos, j, patch_size)
    for k in range(true_pos.shape[0]):
        x_true[k] = gain_neighborhood_pixel(mirror_data, true_pos, k, patch_size)
    print("x_train shape = {}, type = {}".format(x_train.shape, x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape, x_test.dtype))
    print("x_true  shape = {}, type = {}".format(x_true.shape, x_test.dtype))
    print("**************************************************")
    return x_train, x_test, x_true


def train_and_test_label(number_train, number_test, number_true, num_classes):
    y_train = []
    y_test = []
    y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
        for n in range(number_true[i]):
            y_true.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_true = np.array(y_true)
    print("y_train: shape = {}, type = {}".format(y_train.shape,y_train.dtype))
    print("y_test: shape = {}, type = {}".format(y_test.shape,y_test.dtype))
    print("y_true: shape = {}, type = {}".format(y_true.shape,y_true.dtype))
    print("**************************************************")
    return y_train, y_test, y_true