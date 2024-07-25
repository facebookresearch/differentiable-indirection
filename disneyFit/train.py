# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sys
sys.path.insert(1, '../')
import utility as ut
import torch.optim as optim
from networks import *
from shade import *
import os
import imageTools as it
import brdf as bdf
import matplotlib.pyplot as plt
import time

###traincmd: python .\train.py 16 16 16

baseName = "disneyFitTrain"
tab0Res = int(ut.getSysArgv(1))
tab1Res = int(ut.getSysArgv(2)) 
tab2Res = int(ut.getSysArgv(3))

experimentName = baseName + "_tab0_" + str(tab0Res) + "_tab1_" + str(tab1Res) + "_tab2_" + str(tab2Res)

outputDirectory = ut.getOutputDirectory(experimentName) 

torchDevice = ut.getTorchDevice("cuda")

GBUF_EVAL_FILE = "in/gbuffer-eval.npz"

def train():
    batchResolution = 1024

    # Load gbuffer data to render sample images if provided
    if os.path.exists(GBUF_EVAL_FILE):
        gBufData = np.load(GBUF_EVAL_FILE)
    else:
        gBufData = None
    
    disneyNet = DisneyNet(torchDevice, tab0Res, tab1Res, tab2Res)
    optimizableParams = list(disneyNet.parameters())
    optimizer = optim.Adam(optimizableParams, lr=0.0005 * 4, weight_decay=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1000, verbose=True)

    mvAvgErrFrvCTerm = ut.NpQueue(10)
    mvAvgErrFrvMTerm = ut.NpQueue(10)
    mvAvgErrFrsCTerm = ut.NpQueue(10)
    mvAvgErrFcTerm = ut.NpQueue(10)
    mvAvgErrDiffTerm = ut.NpQueue(10)
    mvAvgErrCoefTerm = ut.NpQueue(10)
  
    errorLog = []
    l1LossFn = torch.nn.L1Loss()
    retList = [0, 1, 2, 3, 4, 5, 6, 7]
    retIdx = 0
    startTime = time.time()
    for epoch in range(12050):
        V = ut.uniformSphereSample(torch.rand(size=(batchResolution**2, 2), dtype=torch.float32, device=torchDevice))
        L = ut.uniformSphereSample(torch.rand(size=(batchResolution**2, 2), dtype=torch.float32, device=torchDevice))
        H = torch.abs(ut.normNd(V + L))
         
        roughNoh = triangleWave(torch.stack((torch.rand(size=(batchResolution**2, 1), dtype=torch.float32, device=torchDevice)[:,0], H[:,2]), dim=1)) # triangle(torch.stack((bdf.rougnessLogSample(batchResolution**2).to(torchDevice), H[:,2]), dim=1))
        anisoToh = triangleWave(torch.stack((torch.rand(size=(batchResolution**2, 1), dtype=torch.float32, device=torchDevice)[:,0], H[:,0]), dim=1))
        nDotLV = triangleWave(torch.stack((L[:,2], V[:,2]), dim=1))
        tDotLV = triangleWave(torch.stack((L[:,0], V[:,0]), dim=1))
        metalLoh = triangleWave(torch.stack((torch.rand(size=(batchResolution**2, 1), dtype=torch.float32, device=torchDevice)[:,0] , torch.abs(ut.dotNd(L, H))), dim=1))
        cg = 1 - torch.pow(torch.rand(size=(batchResolution**2, 1), dtype=torch.float32, device=torchDevice)[:,0], 1)
        subCg = triangleWave(torch.stack((torch.rand(size=(batchResolution**2, 1), dtype=torch.float32, device=torchDevice)[:,0], cg), dim=1))
        spst = triangleWave(torch.stack((torch.rand(size=(batchResolution**2, 1), dtype=torch.float32, device=torchDevice)[:,0], torch.rand(size=(batchResolution**2, 1), dtype=torch.float32, device=torchDevice)[:,0]), dim=1))
        shst = triangleWave(torch.stack((torch.rand(size=(batchResolution**2, 1), dtype=torch.float32, device=torchDevice)[:,0], torch.rand(size=(batchResolution**2, 1), dtype=torch.float32, device=torchDevice)[:,0]), dim=1))
        
        albedoNoise = 0.05 + 0.95 * torch.rand(size=(batchResolution**2, 3), dtype=torch.float32, device=torchDevice)
        lum = 0.2126 * albedoNoise[:, 0] + 0.7152 * albedoNoise[:, 1] + 0.0722 * albedoNoise[:, 2]
        lum[lum < 1e-5] = 1
        albedoNoiseTint = albedoNoise / lum[:, None]
        lumCs = triangleWave(torch.stack((lum, torch.rand(size=(batchResolution**2, 1), dtype=torch.float32, device=torchDevice)[:,0]), dim=1))

        disneyNet.train()
        disneyNet.zero_grad()
        tgOut, dTermT, dTermCCT, frvCT, frvMT, diffuseHatT, frsCT, fcT, diffuseT, vTermT, vTermCCT = shadeRef(anisoToh[:, 1], roughNoh[:,1], tDotLV[:, 0], nDotLV[:, 0], tDotLV[:, 1], nDotLV[:, 1], metalLoh[:, 1], roughNoh[:,0], anisoToh[:, 0], subCg[:, 0], subCg[:, 1], metalLoh[:, 0], spst[:, 0], spst[:, 1], shst[:, 0], shst[:, 1], albedoNoise, lumCs[:, 1])
        x, y, dTermN, dTermCCN, frvCN, frvMN, diffuseHatN, frsCN, fcN, err0 = disneyNet(roughNoh, anisoToh, nDotLV, tDotLV, metalLoh, subCg, spst, shst, lumCs, vTermT, vTermCCT, diffuseT)
        netOut = shadeNet(albedoNoiseTint, x, y, dTermN, dTermCCN, frvCN, frvMN, diffuseHatN, frsCN, fcN)[0]
            
        errorDTermL1 = l1LossFn(dTermT, dTermN)
        errorDTermCCL1 = l1LossFn(dTermCCT, dTermCCN)
        errorfrvCTermL1 = l1LossFn(frvCT, frvCN)
        errorfrvMTermL1 = l1LossFn(frvMT, frvMN)
        errorfrsCTermL1 = l1LossFn(frsCT, frsCN)
        errorfcTermL1 = l1LossFn(fcT, fcN)
        errorDiffTermL1 = l1LossFn(diffuseHatT, diffuseHatN)
        
        err0.backward()
        optimizer.step()
                
        with torch.no_grad():
            errorDtermL1Cpu = errorDTermL1.cpu().numpy()
            errorDtermCCL1Cpu = errorDTermCCL1.cpu().numpy()
            errorfrvCTermL1Cpu = errorfrvCTermL1.cpu().numpy()
            errorfrvMTermL1Cpu = errorfrvMTermL1.cpu().numpy()
            errorfrsCTermL1Cpu = errorfrsCTermL1.cpu().numpy()
            errorfcTermL1Cpu = errorfcTermL1.cpu().numpy()
            errorDiffTermL1Cpu = errorDiffTermL1.cpu().numpy()
            errorNetCoefL1Cpu = err0.cpu().numpy()
                   
            with torch.no_grad():
                if epoch % (500 * np.clip((512 // batchResolution)**2, 1, 8)) == 0:
                    fileName = outputDirectory + "disney"
                    retIdx = 0
                    deltaTime = (time.time() - startTime) / 60

                    torch.save({"ht_class" : disneyNet.__class__.__name__, "tab0Res" : tab0Res, "tab1Res" : tab1Res, "tab2Res" : tab2Res, "ht_state" : disneyNet.state_dict()}, fileName + ".bin")

                    if gBufData:
                        shNet, shRef = render(torchDevice, disneyNet, gBufData, retList[retIdx])
                        imgFileName = fileName + "_idx" + str(retList[retIdx])
                        it.saveThumbnailCol(imgFileName + "_shNet", it.toUint8(shNet))
                        np.savez_compressed(imgFileName + "_shNet", shNet)
                        it.saveThumbnailCol(imgFileName + "_shRef", it.toUint8(shRef))
                        np.savez_compressed(imgFileName + "_shRef", shRef)
                        
                        psnr = 10 * np.log(1/np.mean((np.clip(shNet,0,1) - np.clip(shRef,0,1) )**2) ) / np.log(10)
                        log_line = (epoch, deltaTime, retIdx, psnr)
                    else:
                        log_line = (epoch, deltaTime, retIdx)

                    errorLog.append(log_line)
                    np.savetxt(outputDirectory + "error.log", np.array(errorLog), fmt="%6.6f", header="Epoch, Time Elapsed (in min), Output ID, PSNR")

                    retIdx = (retIdx + 1) % len(retList)

        mvAvgErrFrvCTerm.add(errorfrvCTermL1Cpu)
        mvAvgErrFrvMTerm.add(errorfrvMTermL1Cpu)
        mvAvgErrFrsCTerm.add(errorfrsCTermL1Cpu)
        mvAvgErrFcTerm.add(errorfcTermL1Cpu)
        mvAvgErrDiffTerm.add(errorDiffTermL1Cpu)
        mvAvgErrCoefTerm.add(errorNetCoefL1Cpu)

        scheduler.step(errorNetCoefL1Cpu)

        if epoch % 10 == 0:
            print("Epoch:" + str(epoch) + " errorNetCoef: " + str(mvAvgErrCoefTerm.mvAvg()) + " errorFRVC: " + str(mvAvgErrFrvCTerm.mvAvg()) + " errorFRVM: " + str(mvAvgErrFrvMTerm.mvAvg()) + " errorDiff: " + str(mvAvgErrDiffTerm.mvAvg()) + " errorFRSC: " + str(mvAvgErrFrsCTerm.mvAvg()) + " errorClear: " + str(mvAvgErrFcTerm.mvAvg()))

train()