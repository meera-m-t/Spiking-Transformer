
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


use_cuda = True

max_epoch = 800


class S1C1Transform:
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

kernels = [ utils.DoGKernel(3,3/9,6/9),
            utils.DoGKernel(3,6/9,3/9),
            utils.DoGKernel(7,7/9,14/9),
            utils.DoGKernel(7,14/9,7/9),
            utils.DoGKernel(13,13/9,26/9),
            utils.DoGKernel(13,26/9,13/9)]
filter = utils.Filter(kernels, padding = 6, thresholds = 50)
s1 = S1C1Transform(filter)



data_root = "data"
MNIST_train = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform = s1))
MNIST_test = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform = s1))
trainset = DataLoader(MNIST_train, batch_size=len(MNIST_train), shuffle=True)
testset= DataLoader(MNIST_test, batch_size=len(MNIST_test), shuffle=True)

# # functions to show an image
# def imshow(img):
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))

    
# # get some images
# dataiter = iter(dataloader)
# images = dataiter.next()
#plt.imshow(np.transpose(images[0].cpu().detach().numpy(), (1, 2, 0)))


mozafari = ViT(6, 30, 10, (15,15),  360, (0.01, -0.0035), (-0.01, 0.0006), 0.4)



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
    
