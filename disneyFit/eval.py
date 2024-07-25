# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sys
sys.path.insert(1, '../')
import utility as ut
import networks
import os
import torch
from shade import render
import imageTools as it

# cmd: python .\eval.py 16 16 16

baseName = "disneyFitTrain"
tab0Res = int(ut.getSysArgv(1))
tab1Res = int(ut.getSysArgv(2)) 
tab2Res = int(ut.getSysArgv(3))

experimentName = baseName + "_tab0_" + str(tab0Res) + "_tab1_" + str(tab1Res) + "_tab2_" + str(tab2Res)

outputDirectory = ut.getOutputDirectory(experimentName)

GBUF_EVAL_FILE = "in/gbuffer-eval.npz"

def infer():
    if not os.path.exists(GBUF_EVAL_FILE):
        raise Exception(f"Gbuffer file `{GBUF_EVAL_FILE}` not found.")

    gBufData = np.load(GBUF_EVAL_FILE)

    if not os.path.exists(outputDirectory + "eval/"):
        os.makedirs(outputDirectory + "eval/")

    networkBlob = torch.load(outputDirectory + "disney.bin")
    torchDevice = ut.getTorchDevice("cuda")

    className = networkBlob["ht_class"]
    networkFunction = getattr(networks, className)
    disneyNetwork = networkFunction(torchDevice, networkBlob["tab0Res"], networkBlob["tab1Res"], networkBlob["tab2Res"])
    disneyNetwork.eval()
    disneyNetwork.load_state_dict(networkBlob["ht_state"])

    retList = [(0, "main"), (1, "aniso_ggx"), (2, "clear_coat"), (3, "metallic_coef0"), (4, "metallic_coef1"), (5, "disney_diffuse"), (6, "sheen_coef"), (7, "clear_coat_coef")]

    with torch.no_grad():
        logStr = ""
        for ret in retList:
            idx = ret[0]
            name = ret[1]
            
            shNet, shRef = render(torchDevice, disneyNetwork, gBufData, idx)

            imgFileName = outputDirectory + "eval/" + name
            it.saveThumbnailCol(imgFileName + "_network", it.toUint8(shNet))
            it.saveThumbnailCol(imgFileName + "_reference", it.toUint8(shRef))

            psnr = 10 * np.log(1/np.mean((np.clip(shNet,0,1) - np.clip(shRef,0,1) )**2) ) / np.log(10)
            logStr += name + ": " + str(psnr) + "\n"
            
            print("Done: " + name)
        
        with open(outputDirectory + "eval/psnr.txt", "w") as text_file:
            text_file.write(logStr)
infer()