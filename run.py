
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from vit_pytorch import ViT
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from SpykeTorch import snn
from SpykeTorch import functional as sf
from SpykeTorch import visualization as vis
from SpykeTorch import utils
from torchvision import transforms


class S1Transform:
    def __init__(self, filter, timesteps = 5):
        self.to_tensor = transforms.ToTensor()
        self.filter = filter
        self.temporal_transform = utils.Intensity2Latency(timesteps)
        self.cnt = 0
    def __call__(self, image):
        if self.cnt % 1000 == 0:
            print(self.cnt)
        self.cnt+=1
        image = self.to_tensor(image) * 255
        image.unsqueeze_(0)
        image = self.filter(image)
        image = sf.local_normalization(image, 8)
        temporal_image = self.temporal_transform(image)
        return temporal_image.sign().byte()

kernels = [ utils.DoGKernel(7,1,2),
            utils.DoGKernel(7,2,1),]
filter = utils.Filter(kernels, padding = 3, thresholds = 50)
s1 = S1Transform(filter)



data_root = "data"
MNIST_train = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform = s1))
MNIST_test = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform = s1))
trainset = DataLoader(MNIST_train, batch_size=len(MNIST_train), shuffle=True)
testset= DataLoader(MNIST_test, batch_size=len(MNIST_test), shuffle=True)

use_cuda = True


max_epoch = 400

mozafari = ViT(2, 3, 10, (15,15), 50, (0.004, -0.003), (-0.004, 0.0005), 0.3)



if use_cuda:
    mozafari.cuda()


# initial adaptive learning rates
apr = mozafari.stdp_lr[0]
anr = mozafari.stdp_lr[1]
app = mozafari.anti_stdp_lr[1]
anp = mozafari.anti_stdp_lr[0]

adaptive_min = 0.2
adaptive_int = 0.8
apr_adapt = ((1.0 - 1.0 / mozafari.number_of_classes) * adaptive_int + adaptive_min) * apr
anr_adapt = ((1.0 - 1.0 / mozafari.number_of_classes) * adaptive_int + adaptive_min) * anr
app_adapt = ((1.0 / mozafari.number_of_classes) * adaptive_int + adaptive_min) * app
anp_adapt = ((1.0 / mozafari.number_of_classes) * adaptive_int + adaptive_min) * anp

# perf
best_train = np.array([0,0,0,0]) # correct, wrong, silence, epoch
best_test = np.array([0,0,0,0]) # correct, wrong, silence, epoch

# train one batch (here a batch contains all data so it is an epoch)
def train(data, target, network):
    network.train()
    perf = np.array([0,0,0]) # correct, wrong, silence
    network.update_dropout()
    for i in range(len(data)):
        data_in = data[i]
        target_in = target[i]
        if use_cuda:
            data_in = data_in.cuda()
            target_in = target_in.cuda()
        d = network(data_in)
        if d != -1:
            if d == target_in:
                perf[0]+=1
                network.reward()
            else:
                perf[1]+=1
                network.punish()
        else:
            perf[2]+=1
    return perf/len(data)

# test one batch (here a batch contains all data so it is an epoch)
def test(data, target, network):
    network.eval()
    perf = np.array([0,0,0]) # correct, wrong, silence
    for i in range(len(data)):
        data_in = data[i]
        target_in = target[i]
        if use_cuda:
            data_in = data_in.cuda()
            target_in = target_in.cuda()
        d = network(data_in)
        if d != -1:
            if d == target_in:
                perf[0]+=1
            else:
                perf[1]+=1
        else:
            perf[2]+=1
    return perf/len(data)

for epoch in range(max_epoch):
    print("Epoch #:", epoch)
    for data, target in trainset:
        perf_train = train(data, target, mozafari)
    if best_train[0] <= perf_train[0]:
        best_train = np.append(perf_train, epoch)
    print("Current Train:", perf_train)
    print("   Best Train:", best_train)
    for data_test, target_test in testset:
        perf_test = test(data_test, target_test, mozafari)
    if best_test[0] <= perf_test[0]:
        best_test = np.append(perf_test, epoch)
        torch.save(mozafari.state_dict(), "saved.net")
    print(" Current Test:", perf_test)
    print("    Best Test:", best_test)

    #update adaptive learning rates
    apr_adapt = apr * (perf_train[1] * adaptive_int + adaptive_min)
    anr_adapt = anr * (perf_train[1] * adaptive_int + adaptive_min)
    app_adapt = app * (perf_train[0] * adaptive_int + adaptive_min)
    anp_adapt = anp * (perf_train[0] * adaptive_int + adaptive_min)
    mozafari.update_learning_rates(apr_adapt, anr_adapt, app_adapt, anp_adapt)
    
# Features #
feature = torch.tensor([
    [
        [1]
    ]
    ]).float()
if use_cuda:
    feature = feature.cuda()

cstride = (1,1)

# S1 Features #
if use_cuda:
    feature,cstride = vis.get_deep_feature(feature, cstride, (filter.max_window_size, filter.max_window_size), (1,1), filter.kernels.cuda())
else:
    feature,cstride = vis.get_deep_feature(feature, cstride, (filter.max_window_size, filter.max_window_size), (1,1), filter.kernels)
# C1 Features #
feature,cstride = vis.get_deep_feature(feature, cstride, (s1.pooling_size, s1.pooling_size), (s1.pooling_stride, s1.pooling_stride))
# S2 Features #
feature,cstride = vis.get_deep_feature(feature, cstride, mozafari.kernel_size, (1,1), mozafari.s2.weight)

for i in range(mozafari.number_of_features):
    vis.plot_tensor_in_image('feature_s2_'+str(i).zfill(4)+'.png',feature[i])
