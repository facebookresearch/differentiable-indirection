# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sys
sys.path.insert(1, '../')
import utility as ut
import networks
import networksBase as nb
import os
import imageTools as it
import torch
import glob

# cmd: python ./eval.py 6 "Network_p2_c4_401" "6_0"

className = ut.getSysArgv(2)
imageFolder = ut.getSysArgv(3)
experimentName = className
compressionRatio = int(ut.getSysArgv(1))

baseDataDirectory = ut.getOutputDirectory()
experimentName += "_CR" + str(compressionRatio)

imageDirectory = baseDataDirectory + "DifferentiableIndirectionData/imageCache/" + imageFolder
outputDirectory = baseDataDirectory + "DifferentiableIndirectionOutput/" + imageFolder + "/" + experimentName + "/"

assert os.path.exists(outputDirectory), "Trained network directory does not exist."

def getRgbTexList():
    textureNames = []
    textures = []
    textureIdxs = []
    idx = 0
    for filename in os.listdir(imageDirectory):
        f = os.path.join(imageDirectory, filename)
        if os.path.isfile(f):
            rgb = it.readImage(f)
            assert rgb.shape[0] == rgb.shape[1], "Do not support non-square textures"
            assert np.min(rgb) >= 0.0 and np.max(rgb) <= 1.0, "Problem with file " + f
            textureNames.append(filename)
            #plt.imshow(rgb[:,:,:3])
            #plt.show()
            textures.append(rgb[:,:,:3])
            textureIdxs.append(idx)
            idx += 1
    
    return textureNames, textures, textureIdxs

def infer():
    if not os.path.exists(outputDirectory + "eval/"):
        os.makedirs(outputDirectory + "eval/")
    torchDevice = ut.getTorchDevice("cpu")

    refImage = torch.from_numpy(getRgbTexList()[1][0]).to(torchDevice) 

    binPath = outputDirectory + "*.bin"
    binFiles = glob.glob(binPath)
    
    networkBlob = torch.load(binFiles[0])
    networkName = networkBlob["ht_class"]
    networkFunction = getattr(networks, networkName)
    assert className == networkName
    assert compressionRatio == networkBlob["ht_compressionRatioExpected"]
    network = networkFunction(torchDevice, networkBlob["ht_resNative"], compressionRatio)
    network.load_state_dict(networkBlob["ht_state"])
    network.precompute()
    network.eval()

    with torch.no_grad():
        uvSamples = ut.generateUvCoord(torchDevice, networkBlob["ht_resNative"], networkBlob["ht_resNative"]).reshape((-1,2))
        networkOutput = network.infer(uvSamples).reshape((networkBlob["ht_resNative"], networkBlob["ht_resNative"], -1))
        reference = networks.bilinear2d(uvSamples, refImage)[0].reshape((networkBlob["ht_resNative"], networkBlob["ht_resNative"], -1))
        mseErr = torch.nn.functional.mse_loss(networkOutput, reference).cpu().numpy()
        psnr = 10 * np.log(1/mseErr) / np.log(10)
        psnrStr = "PSNR: " + str(psnr)
        it.saveThumbnailCol(outputDirectory + "eval/networkOutput", it.toUint8(networkOutput.cpu().numpy()[::-1]))

        with open(outputDirectory + "eval/psnr.txt", "w") as text_file:
            text_file.write(psnrStr)
infer()