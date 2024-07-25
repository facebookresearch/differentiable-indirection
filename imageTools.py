# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import PIL
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import matplotlib.pyplot as plt

def toUint8(inp):
    return (255.01 * np.clip(inp, 0, 1)).round().astype(np.uint8)

def saveThumbnailCol(name, arr, ext=".jpg"):
    im = Image.fromarray(arr)
    im.save(name + ext)

def readImg(fileName):
    f = np.asarray(PIL.Image.open(fileName))
    return f.astype(np.float32) / 255.0

def readImgUint(fileName):
    f = np.asarray(PIL.Image.open(fileName))
    return f

def rgba2rgb(rgba, background=(1.0,1.0,1.0)):
    row, col, ch = rgba.shape
    
    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' )

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return rgb

def imgResize(torchImg, size):
    res = (torchImg.shape[0] // size + 1) * size
   
    while (res >= size):
        uvList = ut.generateUvCoord(torchImg.device, res, res).reshape((-1,2))
        torchImg = torch.flip(bilinear2d(uvList, torchImg)[0].reshape((res, res, 3)), [0])

        res = res // 2
        if res < size:
            uvList = ut.generateUvCoord(torchImg.device, size, size).reshape((-1,2))
            torchImg = torch.flip(bilinear2d(uvList, torchImg)[0].reshape((size, size, 3)), [0])

    return torchImg

def readImage(fileName):
    if fileName[-3:] == "jpg":
        return rgba2rgb(readImg(fileName))
    elif fileName[-3:] == "png":
        return rgba2rgb(readImg(fileName))
    elif fileName[-3:] == "bmp":
        return rgba2rgb(readImg(fileName))
    elif fileName[-3:] == "hdr":
        return rgba2rgb(readHdr(fileName))
    else:
        print("ReadImage extension not found")
        return None
