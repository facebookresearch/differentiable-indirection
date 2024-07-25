# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sys
sys.path.insert(1, '../')
import utility as ut
import torch.optim as optim
import networks
import torch
import os
import imageTools as it
import time

# traincmd: python ./train.py 48 "Network_p2_c4_401" "6_0"

className = ut.getSysArgv(2)
imageFolder = ut.getSysArgv(3)
experimentName = className
compressionRatio = int(ut.getSysArgv(1))

experimentName += "_CR" + str(compressionRatio)

textureDirectory = f"in/{imageFolder}/"
outputDirectory = ut.getOutputDirectory(experimentName)

torchDevice = ut.getTorchDevice("cuda")

def getRgbTexList():
    textureNames = []
    textures = []
    textureIdxs = []
    idx = 0
    for filename in os.listdir(textureDirectory):
        f = os.path.join(textureDirectory, filename)
        if os.path.isfile(f):
            rgb = it.readImage(f)
            assert rgb.shape[0] == rgb.shape[1], "Only square textures are supported"
            assert np.min(rgb) >= 0.0 and np.max(rgb) <= 1.0, "Problem with file " + f
            textureNames.append(filename)
            #plt.imshow(rgb[:,:,:3])
            #plt.show()
            textures.append(rgb[:,:,:3])
            textureIdxs.append(idx)
            idx += 1
    
    return textureNames, textures, textureIdxs

def train():
    textureNames, textures, textureIdxs = getRgbTexList()
    batchResolution = 1024

    texNets = []
    optimizers = []
    schedulers = []
    torchTargets = []
    uvScaleOffsets = []

    for (texture, textureName) in zip(textures, textureNames):
        resolution = int(np.around(np.sqrt(texture.shape[0] * texture.shape[1])))

        target = torch.from_numpy(texture).half().to(torchDevice)
        torchTargets.append(target)
        netFn = getattr(networks, className)
        texNet = netFn(torchDevice, resolution, compressionRatio, target) #BlockCompress_Arch0
        
        fileName = outputDirectory + ut.getFileNameWoExt(textureName)
        np.savetxt(fileName + "_cr.txt", np.array([texNet.compressionRatio]), fmt="%6.6f")
        np.savetxt(fileName + "_tab0.txt", np.array(texNet.grid0.table.shape), fmt="%6.6f")
        np.savetxt(fileName + "_tab1.txt", np.array(texNet.grid1.table.shape), fmt="%6.6f")
        texNets.append(texNet)
        
        optimizableParams = list(texNet.parameters())
        optimizer = optim.Adam(optimizableParams, lr=0.001, weight_decay=0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=200, verbose=True)
        optimizers.append(optimizer)
        schedulers.append(scheduler)

        assert resolution % batchResolution == 0, "Resolution not multiple of batch size."

        batchSize = np.maximum(int(np.around(resolution / batchResolution)), 1)

        seed = np.arange(batchSize) / batchSize 
        xx, yy = np.meshgrid(seed, seed)
        uvOffset = np.stack((xx.flatten(), yy.flatten()), axis=1)
        uvScaleOffsets.append((torch.from_numpy(uvOffset).to(torch.float).to(torchDevice), batchSize))

    mvAvgErr = ut.NpQueue(10)
    errorLogTrain = []
    errorLogInfer = []
    startTime = time.time()
    for epoch in range(20000):
        uvList = ut.generateUvCoord(torchDevice, batchResolution, batchResolution, True).reshape((-1,2))
        totalError = 0
        for i in range(len(textures)):
            nBatches = uvScaleOffsets[i][1]
            uvOffset = uvScaleOffsets[i][0]
            
            for offsetIdx in range(nBatches * nBatches):
                texNets[i].train()
                uvListOffset = uvList / nBatches + uvOffset[offsetIdx][None,:]
                optimizers[i].zero_grad()

                target = networks.bilinear2d(uvListOffset, torchTargets[i])[0]
                netOut = texNets[i](uvListOffset)
                error = torch.nn.functional.l1_loss(target, netOut) 

                regularization = texNets[i].regularization(epoch)
                (error + regularization).backward()
                texNets[i].disableGrad(epoch)
                optimizers[i].step()

                with torch.no_grad():
                    errorCpu = error.cpu().numpy() / (nBatches * nBatches)
                    totalError += errorCpu
                    
            schedulers[i].step(errorCpu)

            with torch.no_grad():
                if epoch % (200 * np.clip((512 // batchResolution)**2, 1, 8)) == 0:
                    fileName = outputDirectory + ut.getFileNameWoExt(textureNames[i])
                    torch.save({"ht_class" : texNets[i].__class__.__name__, "ht_resNative" : textures[i].shape[0], "ht_compressionRatioActual" : texNets[i].compressionRatio, "ht_compressionRatioExpected" : compressionRatio, "ht_state" : texNets[i].state_dict()}, fileName + ".bin")
                    texNets[i].eval()
                                        
                    uvSamples = ut.generateUvCoord("cpu", textures[i].shape[0], textures[i].shape[1]).reshape((-1,2))
                                                       
                    rgbInterim = torch.zeros((uvSamples.shape[0], textures[i].shape[2]), dtype=torch.float, device="cpu")
                    rgb = torch.zeros(rgbInterim.shape, dtype=torch.float, device="cpu")
                    rgbRef = torch.zeros(rgbInterim.shape, dtype=torch.float, device="cpu")
                    
                    texNets[i].precompute()
                    
                    stepSize = np.minimum(batchResolution, textures[i].shape[0])
                    stepSizeSq = stepSize**2

                    maeAvg = mseAvg = psnrAvg = 0
                    steps = 0
                    for j in range(0, textures[i].shape[0] * textures[i].shape[1], stepSizeSq):
                        uvSamplesDev = uvSamples[j : j + stepSizeSq].to(torchDevice)
                        rgbInterim[j : j + stepSizeSq] = texNets[i](uvSamplesDev)
                        rgbDev = texNets[i].infer(uvSamplesDev)
                        rgbRefDev = networks.bilinear2d(uvSamplesDev, torchTargets[i])[0]
                        rgb[j : j + stepSizeSq]  = rgbDev
                        rgbRef[j : j + stepSizeSq]  = rgbRefDev

                        maeErr = torch.nn.functional.l1_loss(rgbDev, rgbRefDev).cpu().numpy()
                        mseErr = torch.nn.functional.mse_loss(rgbDev, rgbRefDev).cpu().numpy()
                        psnr = 10 * np.log(1/mseErr) / np.log(10)
                        
                        maeAvg += maeErr
                        mseAvg += mseErr
                        psnrAvg += psnr
                        steps += 1

                    psnr = 10 * np.log(steps/mseAvg) / np.log(10)
                    
                    errorLogInfer.append(((time.time() - startTime) / 3600, epoch, maeAvg/steps, mseAvg/steps, psnr))
                    np.savetxt(outputDirectory + "errorInfer.log", np.array(errorLogInfer), fmt="%6.6f", header="Time (in hrs), Epoch, MAE, MSE, PSNR")

                    print("Inference losses -- Mae: " + str(maeAvg/steps) + " Mse: " + str(mseAvg/steps) + " Psnr: " + str(psnr))
                    output = rgb.reshape((textures[i].shape[0], textures[i].shape[1], 3)).numpy()
                    outputInterim = rgbInterim.reshape((textures[i].shape[0], textures[i].shape[1], 3)).numpy()
                    outputRef = rgbRef.reshape((textures[i].shape[0], textures[i].shape[1], 3)).numpy()
                    
                    #np.savez_compressed(fileName, output)
                    it.saveThumbnailCol(fileName, it.toUint8(output[::-1])) #[::-1]
                    it.saveThumbnailCol(fileName + "_interim", it.toUint8(outputInterim[::-1]))
                    it.saveThumbnailCol(fileName + "_ref", it.toUint8(outputRef[::-1]))
            
        mvAvgErr.add(totalError / len(textures))
                
        if epoch % 10 == 0:
            with torch.no_grad():
                print("Training losses -- Epoch: " + str(epoch) + " Mae Error: " + str(mvAvgErr.mvAvg()[0]) + " Regularization: " + str(np.sqrt(regularization[0].cpu().numpy())))
                errorLogTrain.append((epoch, mvAvgErr.mvAvg()[0], np.sqrt(regularization[0].cpu().numpy())))
                np.savetxt(outputDirectory + "errorTrain.log", np.array(errorLogTrain), fmt="%6.6f", header="Epoch, MAE, Regularization")

train()
