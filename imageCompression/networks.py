# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import sys
sys.path.insert(1, '../')
import utility as ut
from networksBase import *

def quantize(input, bits):
    bitMax = (2 ** bits) - 1
    q = torch.round(torch.clamp(input, 0, 1) * bitMax).to(torch.long)
    return torch.clamp(q.to(torch.float) / bitMax, 0, 1-1e-6)

# Primary 2D, Cascaded 2D, rho=4:1 
class Network_p2_c2_41(nn.Module):
    def __init__(self, torchDevice, nativeResolution, expectedCompression, initTarget = None):
        super(Network_p2_c2_41, self).__init__()

        self.Q = 7
        qFactor = self.Q / 8
        self.resolution = nativeResolution * np.sqrt(3 / (expectedCompression * (qFactor * 32 + 3)))
        self.resolution = int(np.around(self.resolution / 4) * 4)
        self.compressionRatio = (3 / (qFactor * 32 + 3)) * ((nativeResolution / self.resolution) ** 2)
        
        print(self.__class__.__name__)
        print("Actual compression ratio: " + str(self.compressionRatio))
        self.grid0 = SpatialGrid2D(torchDevice, uDim=self.resolution, vDim=self.resolution, latent=3, bilinear=True, normalize=True, initScale=1, initMode="R")
        with torch.no_grad():
            if initTarget != None:
                uvInits = ut.generateUvCoord(torchDevice, self.resolution, self.resolution).reshape((-1,2))
                uvInits[:,1] *= -1
                self.grid0.table[:,:,:] = bilinear2d(uvInits, initTarget)[0].reshape((self.resolution, self.resolution, 3))
        self.grid1 = SpatialGrid2D(torchDevice, uDim=self.resolution * 4, vDim=self.resolution * 4, latent=2, bilinear=True, normalize=True, initScale=1, initMode="U")     
 
    def forward(self, x):
        return self.grid0(self.grid1(x))
  
    def precompute(self):
        with torch.no_grad():
            self.grid0Quant = quantize(NON_LINEARITY(self.grid0.table), 8)
            self.grid1Quant =  quantize(NON_LINEARITY(self.grid1.table), self.Q)
    
    def infer(self, x):
        with torch.no_grad():
            key = bilinear2d(x, self.grid1Quant, False)[0]
            return bilinear2d(key, self.grid0Quant, False)[0]

    def disableGrad(self, epoch):
        pass

    def printStuff(self):
        pass

    def regularization(self, epoch):
        return torch.zeros((1), device=self.grid0.table.device)

# Primary 2D, Cascaded 3D, rho=32:1
class Network_p2_c3_321(nn.Module):
    def __init__(self, torchDevice, nativeResolution, expectedCompression, initTarget = None):
        super(Network_p2_c3_321, self).__init__()

        latentIn = 3
        latentGUp = 3
        latentGDn = 3
        
        self.Q = 7
        qFactor = self.Q/8
       
        minErr = 99999999999
        for i in range(nativeResolution):
            testResolution = i + 1
            resGUp = 32 * testResolution
            resGDn = testResolution
           
            lhs = latentIn * (nativeResolution**2) / (qFactor * latentGUp * resGUp**2 + latentGDn * resGDn**3)
            err = np.abs(lhs - expectedCompression)

            if err < minErr:
                self.resolutionDn = testResolution
                minErr = err

        self.resolutionDn = int(np.around(self.resolutionDn / 4) * 4)

        minErr = 99999999999
        for i in range(nativeResolution):
            resGUp = i + 1
            lhs = latentIn * (nativeResolution**2) / (qFactor * latentGUp * resGUp**2 + latentGDn * self.resolutionDn**3)
            err = np.abs(lhs - expectedCompression)

            if err < minErr:
                self.resolutionUp = resGUp
                minErr = err
        
        self.resolutionUp = int(np.around(self.resolutionUp / 8) * 8)
        self.compressionRatio = latentIn * (nativeResolution**2) / (qFactor * latentGUp * self.resolutionUp**2 + latentGDn * self.resolutionDn**3)
        
        print(self.__class__.__name__)
        print("Actual compression ratio:" + str(self.compressionRatio))
        self.grid0 = SpatialGrid3D(torchDevice, uDim=self.resolutionDn, vDim=self.resolutionDn, wDim=self.resolutionDn, latent=latentGDn, bilinear=True, normalize=True, initScale=1, initMode="R")
        self.grid1 = SpatialGrid2D(torchDevice, uDim=self.resolutionUp, vDim=self.resolutionUp, latent=latentGUp, bilinear=True, normalize=True, initScale=1, initMode="U")

    def forward(self, x):
        return self.grid0(self.grid1(x))
    
    def precompute(self):
        with torch.no_grad():
            self.grid0Quant = quantize(NON_LINEARITY(self.grid0.table), 8)
            self.grid1Quant = quantize(NON_LINEARITY(self.grid1.table), self.Q)

    def infer(self, x):
        with torch.no_grad():
            key = bilinear2d(x, self.grid1Quant, False)[0]
            return trilinear3d(key, self.grid0Quant, False)[0]

    def regularization(self, epoch):
        return torch.zeros((1,), device=self.grid1.table.device)
    
    def disableGrad(self, epoch):
        pass

    def printStuff(self):
        pass

# Primary 2D, Cascaded 4D, rho=40:1
class Network_p2_c4_401(nn.Module):
    def __init__(self, torchDevice, nativeResolution, expectedCompression, initTarget = None):
        super(Network_p2_c4_401, self).__init__()

        latentIn = 3
        latentGUp = 4
        latentGDn = 3
        
        self.Q = 7
        qFactor = self.Q/8

        minErr = 99999999999
        for i in range(nativeResolution):
            testResolution = i + 1
            resGUp = 40 * testResolution
            resGDn = testResolution
           
            lhs = latentIn * (nativeResolution**2) / (qFactor * latentGUp * resGUp**2 + latentGDn * resGDn**4)
            err = np.abs(lhs - expectedCompression)

            if err < minErr:
                self.resolutionDn = testResolution
                minErr = err

        self.resolutionDn = int(np.around(self.resolutionDn / 4) * 4)

        minErr = 99999999999
        for i in range(nativeResolution):
            resGUp = i + 1
            lhs = latentIn * (nativeResolution**2) / (qFactor * latentGUp * resGUp**2 + latentGDn * self.resolutionDn**4)
            err = np.abs(lhs - expectedCompression)

            if err < minErr:
                self.resolutionUp = resGUp
                minErr = err
        
        self.resolutionUp = int(np.around(self.resolutionUp / 8) * 8)

        self.compressionRatio = latentIn * (nativeResolution**2) / (qFactor * latentGUp * self.resolutionUp**2 + latentGDn * self.resolutionDn**4)
        
        print(self.__class__.__name__)
        print("Actual compression ratio:" + str(self.compressionRatio))
        self.grid0 = SpatialGrid4D(torchDevice, uDim=self.resolutionDn, vDim=self.resolutionDn, wDim=self.resolutionDn, qDim=self.resolutionDn, latent=latentGDn, bilinear=True, normalize=True, initScale=1, initMode="R")
        self.grid1 = SpatialGrid2D(torchDevice, uDim=self.resolutionUp, vDim=self.resolutionUp, latent=latentGUp, bilinear=True, normalize=True, initScale=1, initMode="U")
    
    def forward(self, x):
        return self.grid0(self.grid1(x))

    def precompute(self):
        with torch.no_grad():
            self.grid0Quant = quantize(NON_LINEARITY(self.grid0.table), 8)
            self.grid1Quant = quantize(NON_LINEARITY(self.grid1.table), self.Q)
    
    def infer(self, x):
        with torch.no_grad():
            key = bilinear2d(x, self.grid1Quant, False)[0]
            return quad4d(key, self.grid0Quant, False)[0]

    def disableGrad(self, epoch):
        pass

    def printStuff(self):
        pass

    def regularization(self, epoch):
        return torch.zeros((1), device=self.grid0.table.device)
