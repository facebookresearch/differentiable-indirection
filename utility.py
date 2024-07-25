# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import sys
import os

FLOAT_MIN = 1e-10

def isCluster():
    if len(sys.argv) > 1 and sys.argv[1] == "CC":
        return True
    
    return False

def getSysArgv(idx):
    if isCluster():
        return sys.argv[idx+1]
    return sys.argv[idx]

def getOutputDirectory(experiment_name):
    output_dir = f"out/{experiment_name}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir

def getFileNameWoExt(fileName):
    base = os.path.basename(fileName)
    return os.path.splitext(base)[0]

def getTorchDevice(name):
    device = torch.device(name)
    if name == "cpu":
        gpuName = "Using CPU device"
    else:
        gpuName = str(torch.cuda.get_device_name(device=device))
    print("Gpu:" + gpuName)

    return device

def roundX(inp, X):
    x = int(np.around(X))
    t = int(np.around(inp / x))

    if t == 0 or t == 1:
        return x
    elif t % 2 == 0:
        return t * x
    
    return (t - 1) * x

# https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations
def uniformSphereSample(uvList):
    ret = torch.zeros((uvList.shape[0], 3), dtype=uvList.dtype, device=uvList.device)

    ret[:, 2] = 2* uvList[:, 0] - 1# cosTheta
    sinTheta = torch.sqrt(1 - ret[:, 2]**2)
    phi = 2 * np.pi * uvList[:, 1]
    ret[:, 0] = torch.cos(phi) * sinTheta
    ret[:, 1] = torch.sin(phi) * sinTheta
    
    return ret

def uniformHemisphereSample(uvList):
    ret = torch.zeros((uvList.shape[0], 3), dtype=uvList.dtype, device=uvList.device)

    ret[:, 2] = uvList[:, 0] # cosTheta
    sinTheta = torch.sqrt(1 - ret[:, 2]**2)
    phi = 2 * np.pi * uvList[:, 1]
    ret[:, 0] = torch.cos(phi) * sinTheta
    ret[:, 1] = torch.sin(phi) * sinTheta
    
    return ret

def uniformHemisphereCosineSample(uvList):
    ret = torch.zeros((uvList.shape[0], 3), dtype=uvList.dtype, device=uvList.device)
    
    ret[:, 2] = torch.sqrt(1 - uvList[:, 0]) # cosTheta
    sinTheta = torch.sqrt(uvList[:, 0])
    phi = 2 * np.pi * uvList[:, 1]
    ret[:, 0] = torch.cos(phi) * sinTheta
    ret[:, 1] = torch.sin(phi) * sinTheta

    return ret

def powerSample(uList, exponent):
    return uList**int(np.around(exponent))

def dotNd(vec1, vec2):
    return torch.sum(vec1 * vec2, dim=1)

def lengthNd(vec):
    return torch.sqrt(dotNd(vec, vec))

def normNd(vec):
    return vec / lengthNd(vec)[:,None]

def dot3d(vec1, vec2):
    return torch.sum(vec1 * vec2, dim=2)

def length3d(vec):
    return torch.sqrt(dot3d(vec, vec))

def norm3d(vec):
    return vec / length3d(vec)[:,:,None]

def generateUvCoord(torchDevice, height, width, rand=False):
    uv = torch.zeros((height, width, 2), dtype=torch.float, device=torchDevice)

    if rand:
        uvRand = torch.rand((height, width, 2), dtype=torch.float, device=torchDevice)
    else:
        uvRand = torch.ones((height, width, 2), dtype=torch.float, device=torchDevice) * 0.5
    
    uvRand[:,:, 0] /= width
    uvRand[:,:, 1] /= -height

    uVals = torch.arange(width) / width
    vVals = 1 - torch.arange(height) / height
    for h in range(height):
        uv[h,:, 0] = uVals
    for w in range(width):
        uv[:,w, 1] = vVals 

    return torch.clamp(uv + uvRand, 0, 1-1e-6)

def generateUCoord(torchDevice, resolution, rand=False):
    u = (torch.arange(resolution) / resolution).to(torch.float).to(torchDevice)

    if rand:
        uRand = torch.rand(resolution, dtype=torch.float, device=torchDevice)
    else:
        uRand = torch.ones(resolution, dtype=torch.float, device=torchDevice) * 0.5
    
    uRand /= resolution
   
    return torch.clamp(u + uRand, 0, 1-1e-6)

class NpQueue:
    def __init__(self, histCount, dim=1):
        self.array = np.zeros((histCount, dim))
        self.ptr = 0
    
    def add(self, newvalue):
        self.array[self.ptr % self.array.shape[0]] = newvalue
        self.ptr += 1

    def mvAvg(self):
        return np.mean(self.array, axis=0)