# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

TRIANGLE_WAVE_SCALE_OFFSET = 1e-6
SINE_WAVE_FREQ = 1

def triangleWave(input):
    return (2 - TRIANGLE_WAVE_SCALE_OFFSET) * torch.abs(input * 0.5 - torch.floor(input * 0.5 + 0.5)) # reduce scaling by a tiny bit to keep the output slightly below 1 

def triangleWaveGrad(input):
    inFl = torch.abs(torch.floor(input)).to(torch.long) & 1
    return (1 - 2 * inFl) * (2 - TRIANGLE_WAVE_SCALE_OFFSET) / 2

class TriangleWave(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ret = triangleWave(input)       
        ctx.save_for_backward(input)
        return ret
    
    @staticmethod
    def backward(ctx, grad_output):
        return triangleWaveGrad(ctx.saved_tensors[0]) * grad_output

def sineWave(input):
    return 0.5 * (1 - TRIANGLE_WAVE_SCALE_OFFSET) * (1 + torch.sin(SINE_WAVE_FREQ * np.pi * input - np.pi / 2))  # reduce scaling by a tiny bit to keep the output slightly below 1 

def sineWaveGrad(input):
    return 0.5 * (1 - TRIANGLE_WAVE_SCALE_OFFSET) * SINE_WAVE_FREQ * np.pi * torch.cos(SINE_WAVE_FREQ * np.pi * input - np.pi / 2)

class SineWave(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ret = sineWave(input)       
        ctx.save_for_backward(input)
        return ret
    
    @staticmethod
    def backward(ctx, grad_output):
        return sineWaveGrad(ctx.saved_tensors[0]) * grad_output

def uvwrap(uvList):
    tw = TriangleWave.apply
    return tw(uvList)

NON_LINEARITY = triangleWave #sineWave
NON_LINEARITY_GRAD = triangleWaveGrad #sineWaveGrad

class DiffRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

def diffround(x):
    dr = DiffRound.apply
    return dr(x)

def near1d(uList, array1D, isNorm = False):
    index = (uList * array1D.shape[0]).to(torch.long)
    a = array1D[index]

    if isNorm:
        a = NON_LINEARITY(a)
    return a, index

def linear1d(uList, array1D, isNorm = False):
    indexFloat = uList * (array1D.shape[0] - 1)
    frac = torch.frac(indexFloat)
    index = indexFloat.to(torch.long)
        
    alpha = frac.reshape((-1,1))
    a = array1D[index]
    b = array1D[index + 1]

    if isNorm:
        a = NON_LINEARITY(a)
        b = NON_LINEARITY(b)
    
    return a *  (1 - alpha) + alpha * b, index, alpha, (b - a) * (array1D.shape[0] - 1)

def uGradNumeric1D(uList, array1D, grad_output, delta, isNorm=False):
    uPlus = triangle(uList + delta)
    uMinius = triangle(uList - delta)
   
    uPosO = linear1d(uPlus, array1D, isNorm)[0]
    uNegO = linear1d(uMinius, array1D, isNorm)[0]
        
    return torch.sum((uPosO - uNegO) * grad_output, dim=1) / (2 * delta)

# class Array1DNearLookup(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, uList, array1D):
#         ret, index = near1d(uList, array1D)
#         ctx.save_for_backward(uList, array1D, index)
#         return ret

#     @staticmethod
#     def backward(ctx, grad_output): # input is dl/do, output is dl/di - need to compute do/di and return its inner product with dl/do
#         uList, array1D, index  = ctx.saved_tensors
#         # gradient w.r.t array content
#         weight_grad = torch.zeros(array1D.shape, device=array1D.device, dtype=array1D.dtype)
#         weight_grad.index_add_(0, index, grad_output)

#         # gradient w.r.t UV
#         u_grad0 = uGradNumeric1D(uList, array1D, grad_output, 1e-3)
#         u_grad1 = uGradNumeric1D(uList, array1D, grad_output, 1e-4)
        
#         return 0.3 * u_grad0 + 0.7 * u_grad1, weight_grad

class Array1DNearLookup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, uList, array1D):
        ret = near1d(uList, array1D)[0]
        ctx.save_for_backward(uList, array1D)
        return ret

    @staticmethod
    def backward(ctx, grad_output): # input is dl/do, output is dl/di - need to compute do/di and return its inner product with dl/do
        uList, array1D  = ctx.saved_tensors
        ret, index, alpha, uGradient = linear1d(uList, array1D)

        # gradient w.r.t array content
        weight_grad = torch.zeros(array1D.shape, device=array1D.device, dtype=array1D.dtype)
        weight_grad.index_add_(0, index, grad_output)

        # gradient w.r.t UV
        #u_grad0 = uGradNumeric1D(uList, array1D, grad_output, 1e-3)
        #u_grad1 = uGradNumeric1D(uList, array1D, grad_output, 1e-4)
       
        return torch.sum(uGradient * grad_output, dim=1), weight_grad #0.3 * u_grad0 + 0.7 * u_grad1

class Array1DNearLookupNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, uList, array1D):
        ret = near1d(uList, array1D, True)[0]
        ctx.save_for_backward(uList, array1D)
        return ret

    @staticmethod
    def backward(ctx, grad_output): # input is dl/do, output is dl/di - need to compute do/di and return its inner product with dl/do
        uList, array1D  = ctx.saved_tensors
        ret, index, alpha, uGradient = linear1d(uList, array1D, True)

        # gradient w.r.t array content
        weight_grad = torch.zeros(array1D.shape, device=array1D.device, dtype=array1D.dtype)
        weight_grad.index_add_(0, index, grad_output)

        # gradient w.r.t UV
        #u_grad0 = uGradNumeric1D(uList, array1D, grad_output, 1e-3, True)
        #u_grad1 = uGradNumeric1D(uList, array1D, grad_output, 1e-4, True)
 
        return torch.sum(uGradient * grad_output, dim=1), weight_grad * NON_LINEARITY_GRAD(array1D) #0.3 * u_grad0 + 0.7 * u_grad1

class Array1DLinearLookup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, uList, array1D):
        ret, index, alpha, uGradient = linear1d(uList, array1D)
        ctx.save_for_backward(uList, array1D, index, alpha, uGradient)
        return ret

    @staticmethod
    def backward(ctx, grad_output): # input is dl/do, output is dl/di - need to compute do/di and return its inner product with dl/do
        uList, array1D, index, alpha, uGradient  = ctx.saved_tensors
        # gradient w.r.t array content
        weight_grad = torch.zeros(array1D.shape, device=array1D.device, dtype=array1D.dtype)
        
        offsets = [0, 1]
        coefs = [(1 - alpha), alpha]
        for i in range(2):
            offset = offsets[i]
            coef = coefs[i]
            weight_grad.index_add_(0, index + offset, coef * grad_output)

        # gradient w.r.t UV
        #u_grad0 = uGrad1D(uList, array1D, grad_output, 1e-3)
        #u_grad1 = uGrad1D(uList, array1D, grad_output, 1e-4)
        
        return torch.sum(uGradient * grad_output, dim=1), weight_grad

class Array1DLinearLookupNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, uList, array1D):
        ret, index, alpha, uGradient = linear1d(uList, array1D, True)
        ctx.save_for_backward(uList, array1D, index, alpha, uGradient)
        return ret

    @staticmethod
    def backward(ctx, grad_output): # input is dl/do, output is dl/di - need to compute do/di and return its inner product with dl/do
        uList, array1D, index, alpha, uGradient  = ctx.saved_tensors
        # gradient w.r.t array content
        weight_grad = torch.zeros(array1D.shape, device=array1D.device, dtype=array1D.dtype)
        
        offsets = [0, 1]
        coefs = [(1 - alpha), alpha]
        for i in range(2):
            offset = offsets[i]
            coef = coefs[i]
            weight_grad.index_add_(0, index + offset, coef * grad_output)

        # gradient w.r.t UV
        #u_grad0 = uGrad1D(uList, array1D, grad_output, 1e-3, True)
        #u_grad1 = uGrad1D(uList, array1D, grad_output, 1e-6, True)
 
        return torch.sum(uGradient * grad_output, dim=1), weight_grad * NON_LINEARITY_GRAD(array1D) # 0.3 * u_grad0 + 0.7 * u_grad1

def quantizerLookup(x, quantizer):
    op = torch.zeros(x.shape, dtype=torch.float, requires_grad=False, device=x.device)
    for ch in range(x.shape[1]):
        op[:, ch] += quantizer(x[:, ch])[:, 0]
    return op

def quantizerSint8(x):
    return diffround((2 * x - 1) * 127) / 128

def quantizerSint4(x):
    return diffround((2 * x - 1) * 7) / 8

def near2dqs(uvList, array2D, quantizerFn):
    index = (uvList * torch.tensor([array2D.shape[1], array2D.shape[0]]).to(uvList.device)).to(torch.long)
    a = quantizerFn(uvwrap(array2D[index[:,1], index[:,0]]))
    
    return a

def near2dq(uvList, array2D, quantizer):
    index = (uvList * torch.tensor([array2D.shape[1], array2D.shape[0]]).to(uvList.device)).to(torch.long)
    a = quantizerLookup(uvwrap(array2D[index[:,1], index[:,0]]), quantizer)
    
    return a

def near2d(uvList, array2D, isNorm = False):
    index = (uvList * torch.tensor([array2D.shape[1], array2D.shape[0]]).to(uvList.device)).to(torch.long)
    a = array2D[index[:,1], index[:,0]].to(torch.float)
    if isNorm:
        a = NON_LINEARITY(a)
    return a, index

def bilinear2dqs(uvList, array2D, quantizerFn):
    indexFloat = uvList * torch.tensor([array2D.shape[1] - 1, array2D.shape[0] - 1]).to(uvList.device)
    frac = torch.frac(indexFloat)
    index = indexFloat.to(torch.long)
    
    alpha = frac[:,0].reshape((-1,1))
    beta = frac[:,1].reshape((-1,1))

    a = quantizerFn(uvwrap(array2D[index[:,1], index[:,0]]))
    b = quantizerFn(uvwrap(array2D[index[:,1], index[:,0] + 1]))
    c = quantizerFn(uvwrap(array2D[index[:,1] + 1, index[:,0]]))
    d = quantizerFn(uvwrap(array2D[index[:,1] + 1, index[:,0] + 1]))

    p = a *  (1 - alpha) + alpha * b # first row
    q = c *  (1 - alpha) + alpha * d # second row
   
    return p * (1 - beta) + beta * q

def bilinear2dq(uvList, array2D, quantizer):
    indexFloat = uvList * torch.tensor([array2D.shape[1] - 1, array2D.shape[0] - 1]).to(uvList.device)
    frac = torch.frac(indexFloat)
    index = indexFloat.to(torch.long)
    
    alpha = frac[:,0].reshape((-1,1))
    beta = frac[:,1].reshape((-1,1))

    a = quantizerLookup(uvwrap(array2D[index[:,1], index[:,0]]), quantizer)
    b = quantizerLookup(uvwrap(array2D[index[:,1], index[:,0] + 1]), quantizer)
    c = quantizerLookup(uvwrap(array2D[index[:,1] + 1, index[:,0]]), quantizer)
    d = quantizerLookup(uvwrap(array2D[index[:,1] + 1, index[:,0] + 1]), quantizer)

    p = a *  (1 - alpha) + alpha * b # first row
    q = c *  (1 - alpha) + alpha * d # second row
   
    return p * (1 - beta) + beta * q

def bilinear2d(uvList, array2D, isNorm = False, computeGrad = False):
    indexFloat = uvList * torch.tensor([array2D.shape[1] - 1, array2D.shape[0] - 1]).to(uvList.device)
    frac = torch.frac(indexFloat)
    index = indexFloat.to(torch.long)
    
    alpha = frac[:,0].reshape((-1,1))
    beta = frac[:,1].reshape((-1,1))

    a = array2D[index[:,1], index[:,0]]
    b = array2D[index[:,1], index[:,0] + 1]
    c = array2D[index[:,1] + 1, index[:,0]]
    d = array2D[index[:,1] + 1, index[:,0] + 1]

    if isNorm:
        a = NON_LINEARITY(a)
        b = NON_LINEARITY(b)
        c = NON_LINEARITY(c)
        d = NON_LINEARITY(d)

    p = a *  (1 - alpha) + alpha * b # first row
    q = c *  (1 - alpha) + alpha * d # second row

    uGradient = None
    vGradient = None
    
    if computeGrad:
        uGradient = ((1 - beta) * (b - a) + beta * (d - c)) * (array2D.shape[1] - 1)
        vGradient = (q - p) * (array2D.shape[0] - 1)

    return p * (1 - beta) + beta * q, index, alpha, beta, uGradient, vGradient

def uvGradNumeric2D(uvList, array2D, grad_output, delta, isNorm=False):
    uPlus = triangleWave(uvList + torch.tensor([[delta, 0]], device=uvList.device, dtype=uvList.dtype))
    uMinius = triangleWave(uvList + torch.tensor([[-delta, 0]], device=uvList.device, dtype=uvList.dtype))
    vPlus = triangleWave(uvList + torch.tensor([[0.0, delta]], device=uvList.device, dtype=uvList.dtype))
    vMinus = triangleWave(uvList + torch.tensor([[0.0, -delta]], device=uvList.device, dtype=uvList.dtype))

    uPosO = bilinear2d(uPlus, array2D, isNorm)[0]
    uNegO = bilinear2d(uMinius, array2D, isNorm)[0]
    vPosO = bilinear2d(vPlus, array2D, isNorm)[0]
    vNegO = bilinear2d(vMinus, array2D, isNorm)[0]
    
    uv_grad = torch.zeros(uvList.shape, device=uvList.device, dtype=uvList.dtype)
    uv_grad[:,0] = torch.sum((uPosO - uNegO) * grad_output, dim=1) / (2 * delta)
    uv_grad[:,1] = torch.sum((vPosO - vNegO) * grad_output, dim=1) / (2 * delta)

    return uv_grad

class Array2DNearLookup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, uvList, array2D):
        ret, index = near2d(uvList, array2D)
        ctx.save_for_backward(uvList, array2D, index)
        return ret

    @staticmethod
    def backward(ctx, grad_output): # input is dl/do, output is dl/di - need to compute do/di and return its inner product with dl/do
        uvList, array2D, index  = ctx.saved_tensors
        # gradient w.r.t array content
        weight_grad = torch.zeros(array2D.shape, device=array2D.device, dtype=array2D.dtype)
       
        weight_grad_flat = weight_grad.view(-1, weight_grad.shape[-1])
        index_flat = index[:,1] * weight_grad.shape[1] + index[:,0]
        weight_grad_flat.index_add_(0, index_flat, grad_output)

        # gradient w.r.t UV
        uv_grad0 = uvGradNumeric2D(uvList, array2D, grad_output, 1e-3)
        uv_grad1 = uvGradNumeric2D(uvList, array2D, grad_output, 1e-4)
 
        return 0.3 * uv_grad0 + 0.7 * uv_grad1, weight_grad

class Array2DNearLookupNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, uvList, array2D):
        ret, index = near2d(uvList, array2D, True)
        ctx.save_for_backward(uvList, array2D, index)
        return ret

    @staticmethod
    def backward(ctx, grad_output): # input is dl/do, output is dl/di - need to compute do/di and return its inner product with dl/do
        uvList, array2D, index  = ctx.saved_tensors
        # gradient w.r.t array content
        weight_grad = torch.zeros(array2D.shape, device=array2D.device, dtype=array2D.dtype)
       
        weight_grad_flat = weight_grad.view(-1, weight_grad.shape[-1])
        index_flat = index[:,1] * weight_grad.shape[1] + index[:,0]
        weight_grad_flat.index_add_(0, index_flat, grad_output)

        # gradient w.r.t UV
        uv_grad0 = uvGradNumeric2D(uvList, array2D, grad_output, 1e-3, True)
        uv_grad1 = uvGradNumeric2D(uvList, array2D, grad_output, 1e-4, True)
 
        return 0.3 * uv_grad0 + 0.7 * uv_grad1, weight_grad * NON_LINEARITY_GRAD(array2D)

class Array2DBilinearLookup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, uvList, array2D):
        ret = bilinear2d(uvList, array2D, False)[0]
        ctx.save_for_backward(uvList, array2D)
        return ret

    @staticmethod
    def backward(ctx, grad_output): # input is dl/do, output is dl/di - need to compute do/di and return its inner product with dl/do
        uvList, array2D = ctx.saved_tensors
        ret, index, alpha, beta, uGradient, vGradient = bilinear2d(uvList, array2D, False, True)
        # gradient w.r.t array content
        weight_grad = torch.zeros(array2D.shape, device=array2D.device, dtype=array2D.dtype)
              
        # Atomic adds aren't working with this approach
        # weight_grad[index[:,1], index[:,0]] += (1 - alpha) * (1 - beta) * grad_output
        # weight_grad[index[:,1], index[:,0] + 1] += alpha * (1 - beta) * grad_output
        # weight_grad[index[:,1] + 1, index[:,0]] += (1 - alpha) * beta * grad_output
        # weight_grad[index[:,1] + 1, index[:,0] + 1] += alpha * beta * grad_output

        weight_grad_flat = weight_grad.view(-1, weight_grad.shape[-1])
        offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]
        coefs = [(1 - alpha) * (1 - beta), alpha * (1 - beta), (1 - alpha) * beta, alpha * beta]
        for i in range(4):
            offset = offsets[i]
            coef = coefs[i]
            index_flat = (index[:,1] + offset[0]) * weight_grad.shape[1] + index[:,0] + offset[1]
            weight_grad_flat.index_add_(0, index_flat, coef * grad_output)

        # gradient w.r.t UV
        #uv_grad0 = uvGrad2D(uvList, array2D, grad_output, 1e-3)
        #uv_grad1 = uvGrad2D(uvList, array2D, grad_output, 1e-4)

        uv_grad = torch.zeros(uvList.shape, device=uvList.device, dtype=uvList.dtype)
        uv_grad[:,0] = torch.sum(uGradient * grad_output, dim=1)
        uv_grad[:,1] = torch.sum(vGradient * grad_output, dim=1)

        return uv_grad, weight_grad

class Array2DBilinearLookupNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, uvList, array2D):
        ret = bilinear2d(uvList, array2D, True)[0]
        ctx.save_for_backward(uvList, array2D)
        return ret

    @staticmethod
    def backward(ctx, grad_output): # input is dl/do, output is dl/di - need to compute do/di and return its inner product with dl/do
        uvList, array2D = ctx.saved_tensors
        ret, index, alpha, beta, uGradient, vGradient = bilinear2d(uvList, array2D, True, True)
        # gradient w.r.t array content
        weight_grad = torch.zeros(array2D.shape, device=array2D.device, dtype=array2D.dtype)    
       
        weight_grad_flat = weight_grad.view(-1, weight_grad.shape[-1])
        offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]
        coefs = [(1 - alpha) * (1 - beta), alpha * (1 - beta), (1 - alpha) * beta, alpha * beta]
        for i in range(4):
            offset = offsets[i]
            coef = coefs[i]
            index_flat = (index[:,1] + offset[0]) * weight_grad.shape[1] + index[:,0] + offset[1]
            weight_grad_flat.index_add_(0, index_flat, coef * grad_output)

        # gradient w.r.t UV
        #uv_grad0 = uvGrad2D(uvList, array2D, grad_output, 1e-3, True)
        #uv_grad1 = uvGrad2D(uvList, array2D, grad_output, 1e-4, True)

        uv_grad = torch.zeros(uvList.shape, device=uvList.device, dtype=uvList.dtype)
        uv_grad[:,0] = torch.sum(uGradient * grad_output, dim=1)
        uv_grad[:,1] = torch.sum(vGradient * grad_output, dim=1)

        return uv_grad, weight_grad * NON_LINEARITY_GRAD(array2D)

def near3d(uvwList, array3D, isNorm = False):
    index = torch.round((uvwList * torch.tensor([array3D.shape[2], array3D.shape[1], array3D.shape[0]]).to(uvwList.device) )).to(torch.long)
    a = array3D[index[:,2], index[:,1], index[:,0]]
    if isNorm:
        a = NON_LINEARITY(a)
    return a, index

def trilinear3d(uvwList, array3D, isNorm = False, computeGrad = False):
    indexFloat = uvwList * torch.tensor([array3D.shape[2] - 1, array3D.shape[1] - 1, array3D.shape[0] - 1]).to(uvwList.device)
    frac = torch.frac(indexFloat)
    index = indexFloat.to(torch.long)
    
    alpha = frac[:,0].reshape((-1,1))
    beta = frac[:,1].reshape((-1,1))
    gamma = frac[:,2].reshape((-1,1))

    x000 = array3D[index[:,2], index[:,1], index[:,0]]
    x001 = array3D[index[:,2], index[:,1], index[:,0] + 1]
    x010 = array3D[index[:,2], index[:,1] + 1, index[:,0]]
    x011 = array3D[index[:,2], index[:,1] + 1, index[:,0] + 1]

    x100 = array3D[index[:,2] + 1, index[:,1], index[:,0]]
    x101 = array3D[index[:,2] + 1, index[:,1], index[:,0] + 1]
    x110 = array3D[index[:,2] + 1, index[:,1] + 1, index[:,0]]
    x111 = array3D[index[:,2] + 1, index[:,1] + 1, index[:,0] + 1]

    if isNorm:
        x000 = NON_LINEARITY(x000)
        x001 = NON_LINEARITY(x001)
        x010 = NON_LINEARITY(x010)
        x011 = NON_LINEARITY(x011)
        
        x100 = NON_LINEARITY(x100)
        x101 = NON_LINEARITY(x101)
        x110 = NON_LINEARITY(x110)
        x111 = NON_LINEARITY(x111)

    x00 = x000 *  (1 - alpha) + alpha * x001
    x01 = x010 *  (1 - alpha) + alpha * x011

    x10 = x100 *  (1 - alpha) + alpha * x101
    x11 = x110 *  (1 - alpha) + alpha * x111

    x0 = x00 * (1 - beta) + x01 * beta
    x1 = x10 * (1 - beta) + x11 * beta

    x = x0 * (1 - gamma) + x1 * gamma

    uGradient = None
    vGradient = None
    wGradient = None

    if computeGrad:
        uGradient0 = ((1 - beta) * (x001 - x000) + beta * (x011 - x010))
        uGradient1 = ((1 - beta) * (x101 - x100) + beta * (x111 - x110))
        uGradient = ((1 - gamma) * uGradient0 + gamma * uGradient1) * (array3D.shape[2] - 1)
        vGradient = ((1 - gamma) * (x01 - x00) + gamma * (x11 - x10)) * (array3D.shape[1] - 1)
        wGradient = (x1 - x0) * (array3D.shape[0] - 1)

    return x, index, alpha, beta, gamma, uGradient, vGradient, wGradient

class Array3DTrilinearLookup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, uvwList, array3D):
        ret = trilinear3d(uvwList, array3D, False)[0]
        ctx.save_for_backward(uvwList, array3D)
        return ret

    @staticmethod
    def backward(ctx, grad_output): # input is dl/do, output is dl/di - need to compute do/di and return its inner product with dl/do
        uvwList, array3D = ctx.saved_tensors
        ret, index, alpha, beta, gamma, uGradient, vGradient, wGradient = trilinear3d(uvwList, array3D, False, True)
        # gradient w.r.t array content
        weight_grad = torch.zeros(array3D.shape, device=array3D.device, dtype=array3D.dtype)
              
        # Atomic adds aren't working with this approach
        # weight_grad[index[:,1], index[:,0]] += (1 - alpha) * (1 - beta) * grad_output
        # weight_grad[index[:,1], index[:,0] + 1] += alpha * (1 - beta) * grad_output
        # weight_grad[index[:,1] + 1, index[:,0]] += (1 - alpha) * beta * grad_output
        # weight_grad[index[:,1] + 1, index[:,0] + 1] += alpha * beta * grad_output

        weight_grad_flat = weight_grad.view(-1, weight_grad.shape[-1])
        offsets = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
        coefs = [(1 - gamma) * (1 - alpha) * (1 - beta), (1 - gamma) * alpha * (1 - beta), (1 - gamma) * (1 - alpha) * beta, (1 - gamma) * alpha * beta, gamma * (1 - alpha) * (1 - beta), gamma * alpha * (1 - beta), gamma * (1 - alpha) * beta, gamma * alpha * beta]
        for i in range(8):
            offset = offsets[i]
            coef = coefs[i]
            index_flat = (index[:,2] + offset[0]) * weight_grad.shape[2] * weight_grad.shape[1] + (index[:,1] + offset[1]) * weight_grad.shape[2] + index[:,0] + offset[2]
            weight_grad_flat.index_add_(0, index_flat, coef * grad_output)

        uv_grad = torch.zeros(uvwList.shape, device=uvwList.device, dtype=uvwList.dtype)
        uv_grad[:,0] = torch.sum(uGradient * grad_output, dim=1)
        uv_grad[:,1] = torch.sum(vGradient * grad_output, dim=1)
        uv_grad[:,2] = torch.sum(wGradient * grad_output, dim=1)

        return uv_grad, weight_grad

class Array3DTrilinearLookupNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, uvwList, array3D):
        ret = trilinear3d(uvwList, array3D, True)[0]
        ctx.save_for_backward(uvwList, array3D)
        return ret

    @staticmethod
    def backward(ctx, grad_output): # input is dl/do, output is dl/di - need to compute do/di and return its inner product with dl/do
        uvwList, array3D = ctx.saved_tensors
        ret, index, alpha, beta, gamma, uGradient, vGradient, wGradient = trilinear3d(uvwList, array3D, True, True)
        # gradient w.r.t array content
        weight_grad = torch.zeros(array3D.shape, device=array3D.device, dtype=array3D.dtype)
              
        # Atomic adds aren't working with this approach
        # weight_grad[index[:,1], index[:,0]] += (1 - alpha) * (1 - beta) * grad_output
        # weight_grad[index[:,1], index[:,0] + 1] += alpha * (1 - beta) * grad_output
        # weight_grad[index[:,1] + 1, index[:,0]] += (1 - alpha) * beta * grad_output
        # weight_grad[index[:,1] + 1, index[:,0] + 1] += alpha * beta * grad_output

        weight_grad_flat = weight_grad.view(-1, weight_grad.shape[-1])
        offsets = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
        coefs = [(1 - gamma) * (1 - alpha) * (1 - beta), (1 - gamma) * alpha * (1 - beta), (1 - gamma) * (1 - alpha) * beta, (1 - gamma) * alpha * beta, gamma * (1 - alpha) * (1 - beta), gamma * alpha * (1 - beta), gamma * (1 - alpha) * beta, gamma * alpha * beta]
        for i in range(8):
            offset = offsets[i]
            coef = coefs[i]
            index_flat = (index[:,2] + offset[0]) * weight_grad.shape[2] * weight_grad.shape[1] + (index[:,1] + offset[1]) * weight_grad.shape[2] + index[:,0] + offset[2]
            weight_grad_flat.index_add_(0, index_flat, coef * grad_output)

        uv_grad = torch.zeros(uvwList.shape, device=uvwList.device, dtype=uvwList.dtype)
        uv_grad[:,0] = torch.sum(uGradient * grad_output, dim=1)
        uv_grad[:,1] = torch.sum(vGradient * grad_output, dim=1)
        uv_grad[:,2] = torch.sum(wGradient * grad_output, dim=1)

        return uv_grad, weight_grad * NON_LINEARITY_GRAD(array3D)

def quad4d(uvwqList, array4D, isNorm = False, computeGrad = False):
    indexFloat = uvwqList * torch.tensor([array4D.shape[3] - 1, array4D.shape[2] - 1, array4D.shape[1] - 1, array4D.shape[0] - 1]).to(uvwqList.device)
    frac = torch.frac(indexFloat)
    index = indexFloat.to(torch.long)
    
    alpha = frac[:,0].reshape((-1,1))
    beta = frac[:,1].reshape((-1,1))
    gamma = frac[:,2].reshape((-1,1))
    delta = frac[:,3].reshape((-1,1))

    #array4d_Flat = array4D.view(-1, array4D.shape[-1])
    #(index[:,3] + offset[0]) * weight_grad.shape[3] * weight_grad.shape[2] * weight_grad.shape[1]  + (index[:,2] + offset[1]) * weight_grad.shape[3] * weight_grad.shape[2] + (index[:,1] + offset[2]) * weight_grad.shape[3] + index[:,0] + offset[3]

    x0000 = array4D[index[:,3], index[:,2], index[:,1], index[:,0]]
    x0001 = array4D[index[:,3], index[:,2], index[:,1], index[:,0] + 1]
    x0010 = array4D[index[:,3], index[:,2], index[:,1] + 1, index[:,0]]
    x0011 = array4D[index[:,3], index[:,2], index[:,1] + 1, index[:,0] + 1]

    x0100 = array4D[index[:,3], index[:,2] + 1, index[:,1], index[:,0]]
    x0101 = array4D[index[:,3], index[:,2] + 1, index[:,1], index[:,0] + 1]
    x0110 = array4D[index[:,3], index[:,2] + 1, index[:,1] + 1, index[:,0]]
    x0111 = array4D[index[:,3], index[:,2] + 1, index[:,1] + 1, index[:,0] + 1]

    x1000 = array4D[index[:,3] + 1, index[:,2], index[:,1], index[:,0]]
    x1001 = array4D[index[:,3] + 1, index[:,2], index[:,1], index[:,0] + 1]
    x1010 = array4D[index[:,3] + 1, index[:,2], index[:,1] + 1, index[:,0]]
    x1011 = array4D[index[:,3] + 1, index[:,2], index[:,1] + 1, index[:,0] + 1]

    x1100 = array4D[index[:,3] + 1, index[:,2] + 1, index[:,1], index[:,0]]
    x1101 = array4D[index[:,3] + 1, index[:,2] + 1, index[:,1], index[:,0] + 1]
    x1110 = array4D[index[:,3] + 1, index[:,2] + 1, index[:,1] + 1, index[:,0]]
    x1111 = array4D[index[:,3] + 1, index[:,2] + 1, index[:,1] + 1, index[:,0] + 1]  

    if isNorm:
        x0000 = NON_LINEARITY(x0000)
        x0001 = NON_LINEARITY(x0001)
        x0010 = NON_LINEARITY(x0010)
        x0011 = NON_LINEARITY(x0011)
        
        x0100 = NON_LINEARITY(x0100)
        x0101 = NON_LINEARITY(x0101)
        x0110 = NON_LINEARITY(x0110)
        x0111 = NON_LINEARITY(x0111)

        x1000 = NON_LINEARITY(x1000)
        x1001 = NON_LINEARITY(x1001)
        x1010 = NON_LINEARITY(x1010)
        x1011 = NON_LINEARITY(x1011)
        
        x1100 = NON_LINEARITY(x1100)
        x1101 = NON_LINEARITY(x1101)
        x1110 = NON_LINEARITY(x1110)
        x1111 = NON_LINEARITY(x1111)

    x000 = x0000 *  (1 - alpha) + alpha * x0001
    x001 = x0010 *  (1 - alpha) + alpha * x0011

    x010 = x0100 *  (1 - alpha) + alpha * x0101
    x011 = x0110 *  (1 - alpha) + alpha * x0111

    x100 = x1000 *  (1 - alpha) + alpha * x1001
    x101 = x1010 *  (1 - alpha) + alpha * x1011

    x110 = x1100 *  (1 - alpha) + alpha * x1101
    x111 = x1110 *  (1 - alpha) + alpha * x1111
    
    x00 = x000 * (1 - beta) + x001 * beta
    x01 = x010 * (1 - beta) + x011 * beta

    x10 = x100 * (1 - beta) + x101 * beta
    x11 = x110 * (1 - beta) + x111 * beta

    x0 = x00 * (1 - gamma) + x01 * gamma
    x1 = x10 * (1 - gamma) + x11 * gamma

    x = x0 * (1 - delta) + x1 * delta

    uGradient = None
    vGradient = None
    wGradient = None
    qGradient = None
    
    if computeGrad:
        uGradient00 = ((1 - beta) * (x0001 - x0000) + beta * (x0011 - x0010))
        uGradient01 = ((1 - beta) * (x0101 - x0100) + beta * (x0111 - x0110))
        uGradient10 = ((1 - beta) * (x1001 - x1000) + beta * (x1011 - x1010))
        uGradient11 = ((1 - beta) * (x1101 - x1100) + beta * (x1111 - x1110))

        uGradient0 = (1 - gamma) * uGradient00 + gamma * uGradient01
        uGradient1 = (1 - gamma) * uGradient10 + gamma * uGradient11

        uGradient = ((1 - delta) * uGradient0 + delta * uGradient1) * (array4D.shape[3] - 1)
        
        vGradient0 = ((1 - gamma) * (x001 - x000) + gamma * (x011 - x010))
        vGradient1 = ((1 - gamma) * (x101 - x100) + gamma * (x111 - x110))

        vGradient = ((1 - delta) * vGradient0 + delta * vGradient1) * (array4D.shape[2] - 1)
        
        wGradient = ((1 - delta) * (x01 - x00) + delta * (x11 - x10)) * (array4D.shape[1] - 1)
        
        qGradient = (x1 - x0) * (array4D.shape[0] - 1)

    return x, index, alpha, beta, gamma, delta, uGradient, vGradient, wGradient, qGradient

class Array4DQlinearLookup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, uvwqList, array4D):
        ret = quad4d(uvwqList, array4D, False)[0]
        ctx.save_for_backward(uvwqList, array4D)
        return ret

    @staticmethod
    def backward(ctx, grad_output): # input is dl/do, output is dl/di - need to compute do/di and return its inner product with dl/do
        uvwqList, array4D = ctx.saved_tensors
        ret, index, alpha, beta, gamma, delta, uGradient, vGradient, wGradient, qGradient = quad4d(uvwqList, array4D, False, True)
        # gradient w.r.t array content
        weight_grad = torch.zeros(array4D.shape, device=array4D.device, dtype=array4D.dtype)
      
        weight_grad_flat = weight_grad.view(-1, weight_grad.shape[-1])
        offsets = [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1), (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 1, 0), (1, 1, 1, 1)]
        coefs = [(1 - delta) * (1 - gamma) * (1 - alpha) * (1 - beta), 
            (1 - delta) * (1 - gamma) * alpha * (1 - beta), 
            (1 - delta) * (1 - gamma) * (1 - alpha) * beta, 
            (1 - delta) * (1 - gamma) * alpha * beta, 
            (1 - delta) * gamma * (1 - alpha) * (1 - beta), 
            (1 - delta) * gamma * alpha * (1 - beta), 
            (1 - delta) * gamma * (1 - alpha) * beta, 
            (1 - delta) * gamma * alpha * beta, 
            delta * (1 - gamma) * (1 - alpha) * (1 - beta), delta * (1 - gamma) * alpha * (1 - beta), 
            delta * (1 - gamma) * (1 - alpha) * beta, 
            delta * (1 - gamma) * alpha * beta, 
            delta * gamma * (1 - alpha) * (1 - beta), 
            delta * gamma * alpha * (1 - beta), 
            delta * gamma * (1 - alpha) * beta, 
            delta * gamma * alpha * beta]

        for i in range(16):
            offset = offsets[i]
            coef = coefs[i]
            index_flat = (index[:,3] + offset[0]) * weight_grad.shape[3] * weight_grad.shape[2] * weight_grad.shape[1]  + (index[:,2] + offset[1]) * weight_grad.shape[3] * weight_grad.shape[2] + (index[:,1] + offset[2]) * weight_grad.shape[3] + index[:,0] + offset[3]
            weight_grad_flat.index_add_(0, index_flat, coef * grad_output)

        uv_grad = torch.zeros(uvwqList.shape, device=uvwqList.device, dtype=uvwqList.dtype)
        uv_grad[:,0] = torch.sum(uGradient * grad_output, dim=1)
        uv_grad[:,1] = torch.sum(vGradient * grad_output, dim=1)
        uv_grad[:,2] = torch.sum(wGradient * grad_output, dim=1)
        uv_grad[:,3] = torch.sum(qGradient * grad_output, dim=1)
        
        return uv_grad, weight_grad

class Array4DQlinearLookupNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, uvwqList, array4D):
        ret = quad4d(uvwqList, array4D, True)[0]
        ctx.save_for_backward(uvwqList, array4D)
        return ret

    @staticmethod
    def backward(ctx, grad_output): # input is dl/do, output is dl/di - need to compute do/di and return its inner product with dl/do
        uvwqList, array4D = ctx.saved_tensors
        ret, index, alpha, beta, gamma, delta, uGradient, vGradient, wGradient, qGradient = quad4d(uvwqList, array4D, True, True)
        # gradient w.r.t array content
        weight_grad = torch.zeros(array4D.shape, device=array4D.device, dtype=array4D.dtype)
      
        weight_grad_flat = weight_grad.view(-1, weight_grad.shape[-1])
        offsets = [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1), (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 1, 0), (1, 1, 1, 1)]
        coefs = [(1 - delta) * (1 - gamma) * (1 - alpha) * (1 - beta), 
            (1 - delta) * (1 - gamma) * alpha * (1 - beta), 
            (1 - delta) * (1 - gamma) * (1 - alpha) * beta, 
            (1 - delta) * (1 - gamma) * alpha * beta, 
            (1 - delta) * gamma * (1 - alpha) * (1 - beta), 
            (1 - delta) * gamma * alpha * (1 - beta), 
            (1 - delta) * gamma * (1 - alpha) * beta, 
            (1 - delta) * gamma * alpha * beta, 
            delta * (1 - gamma) * (1 - alpha) * (1 - beta), delta * (1 - gamma) * alpha * (1 - beta), 
            delta * (1 - gamma) * (1 - alpha) * beta, 
            delta * (1 - gamma) * alpha * beta, 
            delta * gamma * (1 - alpha) * (1 - beta), 
            delta * gamma * alpha * (1 - beta), 
            delta * gamma * (1 - alpha) * beta, 
            delta * gamma * alpha * beta]

        for i in range(16):
            offset = offsets[i]
            coef = coefs[i]
            index_flat = (index[:,3] + offset[0]) * weight_grad.shape[3] * weight_grad.shape[2] * weight_grad.shape[1]  + (index[:,2] + offset[1]) * weight_grad.shape[3] * weight_grad.shape[2] + (index[:,1] + offset[2]) * weight_grad.shape[3] + index[:,0] + offset[3]
            weight_grad_flat.index_add_(0, index_flat, coef * grad_output)

        uv_grad = torch.zeros(uvwqList.shape, device=uvwqList.device, dtype=uvwqList.dtype)
        uv_grad[:,0] = torch.sum(uGradient * grad_output, dim=1)
        uv_grad[:,1] = torch.sum(vGradient * grad_output, dim=1)
        uv_grad[:,2] = torch.sum(wGradient * grad_output, dim=1)
        uv_grad[:,3] = torch.sum(qGradient * grad_output, dim=1)
        
        return uv_grad, weight_grad * NON_LINEARITY_GRAD(array4D)

class SpatialGrid1D(nn.Module):
    def __init__(self, torchDevice, uDim, latent=1, bilinear=True, normalize=False, initScale=1, initMode="R"):
        super(SpatialGrid1D, self).__init__()
        self.resolution = uDim
        self.quantize = False and self.normalize # Only enable during inference, and only if the contents are normalized
        self.bitMax = (2 ** 8) - 1
        
        if bilinear and normalize:
            print("Using autograd Fn: Array1DLinearLookupNorm")
            self.fwdAutograd = Array1DLinearLookupNorm.apply
        elif bilinear and (not normalize):
            print("Using autograd Fn: Array1DLinearLookup")
            self.fwdAutograd = Array1DLinearLookup.apply
        elif (not bilinear) and normalize:
            print("Using autograd Fn: Array1DNearLookupNorm")
            self.fwdAutograd = Array1DNearLookupNorm.apply
        else:
            print("Using autograd Fn: Array1DNearLookup")
            self.fwdAutograd = Array1DNearLookup.apply 

        print("Creating 1D grid with resolution: " + str(self.resolution) + " with channels: " + str(latent) + " o/p Normalization:" + str(normalize))

        if initMode == "U":
            d = 1 / (2 * self.resolution) + np.arange(self.resolution) / self.resolution
            d2 = np.dstack((d, 1 - d))
            grid = np.zeros((self.resolution, latent))

            for i in range(latent):
                grid[:, i] = d2[0, :, i%2]

            self.table = torch.nn.Parameter(torch.tensor(grid, dtype=torch.float, requires_grad=True, device=torchDevice))
        elif initMode == "US" and normalize == False:
            d = 1 / (2 * self.resolution) + np.arange(self.resolution) / self.resolution
            d2 = (2 * np.dstack((d, 1 - d)) - 1) * initScale
            grid = np.zeros((self.resolution, latent))

            for i in range(latent):
                grid[:, i] = d2[0, :, 0]

            self.table = torch.nn.Parameter(torch.tensor(grid, dtype=torch.float, requires_grad=True, device=torchDevice))
        elif initMode == "R":
            self.table = torch.nn.Parameter(torch.rand((self.resolution, latent), dtype=torch.float, requires_grad=True, device=torchDevice) * initScale)
        elif initMode == "C":
            self.table = torch.nn.Parameter(torch.ones((self.resolution, latent), dtype=torch.float, requires_grad=True, device=torchDevice) * initScale)
        else:
            assert False, "Undefined init mode"
            
    def checkGridOccupancy(self, uList):
        h,_,_ = np.histogram(uList, bins=self.resolution)
        return np.sum(h == 0), self.resolution
    
     # quantize and reconstruct
    def quantizeRecon(self, input):
        if self.quantize:
            q = torch.round(input * self.bitMax).to(torch.uint8)
            return q.to(torch.float) / self.bitMax
        return input

    def forward(self, uList):
        return self.fwdAutograd(uList, self.quantizeRecon(self.table))

class SpatialGrid2D(nn.Module):
    def __init__(self, torchDevice, uDim, vDim, latent=1, bilinear=True, normalize=False, initScale=1, initMode="R"): # initMode = "R" : random, "C" : constant, "U" : UV ramp, "Ra" : Radial outward from 0,0
        super(SpatialGrid2D, self).__init__()
        self.dims = (uDim, vDim)
        self.quantize = False and normalize # Only enable during inference, and only if the contents are normalized
        self.bitMax = (2 ** 8) - 1
        
        if bilinear and normalize:
            print("Using autograd Fn: Array2DBilinearLookupNorm")
            self.fwdAutograd = Array2DBilinearLookupNorm.apply
        elif bilinear and (not normalize):
            print("Using autograd Fn: Array2DBilinearLookup")
            self.fwdAutograd = Array2DBilinearLookup.apply       
        elif (not bilinear) and normalize:
            print("Using autograd Fn: Array2DNearLookupNorm")
            self.fwdAutograd = Array2DNearLookupNorm.apply
        else:
            print("Using autograd Fn: Array2DNearLookup")
            self.fwdAutograd = Array2DNearLookup.apply        
                
        print("Creating 2D grid with resolution: " + str(self.dims) + " with channels: " + str(latent) + " o/p Normalization:" + str(normalize))
        if initMode == "U":
            u = 1 / (2 * uDim) + np.arange(uDim) / uDim
            v = 1 / (2 * vDim) + np.arange(vDim) / vDim
            xx, yy = np.meshgrid(u, v)
            uv = np.dstack((xx, yy))

            grid = np.zeros((vDim, uDim, latent))
            for i in range(latent):
                grid[:,:, i] = uv[:,:, i%2]

            self.table = torch.nn.Parameter(torch.tensor(grid, dtype=torch.float, requires_grad=True, device=torchDevice))       
        elif initMode == "R":
            self.table = torch.nn.Parameter(torch.rand((vDim, uDim, latent), dtype=torch.float, requires_grad=True, device=torchDevice) * initScale)
        elif initMode == "C":
            self.table = torch.nn.Parameter(torch.ones((vDim, uDim, latent), dtype=torch.float, requires_grad=True, device=torchDevice) * initScale)
        elif initMode == "Ra":
            ra = np.zeros((vDim, uDim, latent))
            for i in range(vDim):
                for j in range(uDim):
                    y = i / vDim
                    x = j / uDim
                    d = np.sqrt(x**2+y**2)
                    ra[i, j, :] = d
            self.table = torch.nn.Parameter(torch.tensor(ra, dtype=torch.float, requires_grad=True, device=torchDevice)) 
        else:
            assert False, "Undefined init mode"
            
    def checkGridOccupancy(self, uvList):
        h,_,_ = np.histogram2d(uvList[:,0], uvList[:,1], bins=(self.resolution, self.resolution))
        return np.sum(h == 0), self.resolution**2
    
    def quantizeRecon(self, input):
        if self.quantize:
            q = torch.round(input * self.bitMax).to(torch.uint8)
            return q.to(torch.float) / self.bitMax
        return input
    
    def forward(self, uvList):
        return self.fwdAutograd(uvList, self.quantizeRecon(self.table))

class SpatialGrid3D(nn.Module):
    def __init__(self, torchDevice, uDim, vDim, wDim, latent=1, bilinear=True, normalize=False, initScale=1, initMode="R"): # initMode = "R" : random, "C" : constant, "U" : UV ramp
        super(SpatialGrid3D, self).__init__()
        self.dims = (uDim, vDim, wDim)

        if bilinear == False:
            print("Bilinear must be true")
            return
        
        if bilinear and normalize:
            print("Using autograd Fn: Array3DTrilinearLookupNorm")
            self.fwdAutograd = Array3DTrilinearLookupNorm.apply
        elif bilinear and (not normalize):
            print("Using autograd Fn: Array3DTrilinearLookup")
            self.fwdAutograd = Array3DTrilinearLookup.apply       
        else:
            print("Unsupported")    
                
        print("Creating 3D grid with resolution: " + str(self.dims) + " with channels: " + str(latent) + " o/p Normalization:" + str(normalize))
        if initMode == "U":
            u = 1 / (2 * uDim) + np.arange(uDim) / uDim
            v = 1 / (2 * vDim) + np.arange(vDim) / vDim
            w = 1 / (2 * wDim) + np.arange(wDim) / wDim
                       
            uvw =  np.zeros((wDim, vDim, uDim, 3), dtype=np.float32)
         
            for j in range(wDim):
                for k in range(vDim):
                    for l in range(uDim):
                        uvw[j, k, l, 0] = u[l]
                        uvw[j, k, l, 1] = v[k]
                        uvw[j, k, l, 2] = w[j]
                       

            grid = np.zeros((wDim, vDim, uDim, latent), dtype=np.float32)
            for i in range(latent):
                grid[:,:,:, i] = uvw[:,:,:, i%3]

            self.table = torch.nn.Parameter(torch.tensor(grid, dtype=torch.float, requires_grad=True, device=torchDevice))       
        elif initMode == "R":
            self.table = torch.nn.Parameter(torch.rand((wDim, vDim, uDim, latent), dtype=torch.float, requires_grad=True, device=torchDevice) * initScale)
        elif initMode == "C":
            self.table = torch.nn.Parameter(torch.ones((wDim, vDim, uDim, latent), dtype=torch.float, requires_grad=True, device=torchDevice) * initScale)
        else:
            assert False, "Undefined init mode"
       
    def forward(self, uvList):
        return self.fwdAutograd(uvList, self.table)

class SpatialGrid4D(nn.Module):
    def __init__(self, torchDevice, uDim, vDim, wDim, qDim, latent=1, bilinear=True, normalize=False, initScale=1, initMode="R"): # initMode = "R" : random, "C" : constant, "U" : UV ramp
        super(SpatialGrid4D, self).__init__()
        self.dims = (uDim, vDim, wDim, qDim)

        if bilinear == False:
            print("Bilinear must be true")
            return
        
        if bilinear and normalize:
            print("Using autograd Fn: Array4DQlinearLookupNorm")
            self.fwdAutograd = Array4DQlinearLookupNorm.apply
        elif bilinear and (not normalize):
            print("Using autograd Fn: Array4DQlinearLookup")
            self.fwdAutograd = Array4DQlinearLookup.apply       
        else:
            print("Unsupported")    
                
        print("Creating 4D grid with resolution: " + str(self.dims) + " with channels: " + str(latent) + " o/p Normalization:" + str(normalize))
        if initMode == "U":
            u = 1 / (2 * uDim) + np.arange(uDim) / uDim
            v = 1 / (2 * vDim) + np.arange(vDim) / vDim
            w = 1 / (2 * wDim) + np.arange(wDim) / wDim
            q = 1 / (2 * qDim) + np.arange(qDim) / qDim
            
            uvwq =  np.zeros((qDim, wDim, vDim, uDim, 4), dtype=np.float32)

            for i in range(qDim):
                for j in range(wDim):
                    for k in range(vDim):
                        for l in range(uDim):
                            uvwq[i, j, k, l, 0] = u[l]
                            uvwq[i, j, k, l, 1] = v[k]
                            uvwq[i, j, k, l, 2] = w[j]
                            uvwq[i, j, k, l, 3] = q[i]

            grid = np.zeros((qDim, wDim, vDim, uDim, latent), dtype=np.float32)
            for i in range(latent):
                grid[:,:,:,:, i] = uvwq[:,:,:,:, i%4]

            self.table = torch.nn.Parameter(torch.tensor(grid, dtype=torch.float, requires_grad=True, device=torchDevice))       
        elif initMode == "R":
            self.table = torch.nn.Parameter(torch.rand((qDim, wDim, vDim, uDim, latent), dtype=torch.float, requires_grad=True, device=torchDevice) * initScale)
        elif initMode == "C":
            self.table = torch.nn.Parameter(torch.ones((qDim, wDim, vDim, uDim, latent), dtype=torch.float, requires_grad=True, device=torchDevice) * initScale)
        else:
            assert False, "Undefined init mode"
       
    def forward(self, uvList):
        return self.fwdAutograd(uvList, self.table)

class PseudoGridND(nn.Module):
    def __init__(self, torchDevice, dims, latentSpace=1, bilinear=True, normalize=False, initScale=1, initMode="R"): # initMode = "R" : random, "C" : constant, 
        super(PseudoGridND, self).__init__()
        self.dims = dims
                        
        if bilinear and normalize:
            print("Using autograd Fn: PseudoArrayNdLinearLookupNorm")
            self.fwdAutograd = PseudoArrayNdLinearLookupNorm.apply
        elif bilinear and (not normalize):
            print("Using autograd Fn: PseudoArrayNdLinearLookup")
            self.fwdAutograd = PseudoArrayNdLinearLookup.apply
        elif (not bilinear) and normalize:
            print("Using autograd Fn: PseudoArrayNdNearLookupNorm")
            self.fwdAutograd = PseudoArrayNdNearLookupNorm.apply
        else:
            print("Using autograd Fn: PseudoArrayNdNearLookup")
            self.fwdAutograd = PseudoArrayNdNearLookup.apply       
        
        self.dimensions = len(dims)
        print("Creating " + str(self.dimensions) + "D grid with resolution: " + str(self.dims) + " with channels: " + str(latentSpace) + " o/p Normalization:" + str(normalize))
              
        if initMode == "R":
            self.table = torch.nn.Parameter(torch.rand(dims + (latentSpace,), dtype=torch.float, requires_grad=True, device=torchDevice) * initScale)
        elif initMode == "C":
            self.table = torch.nn.Parameter(torch.ones(dims + (latentSpace,), dtype=torch.float, requires_grad=True, device=torchDevice) * initScale)
        else:
            assert False, "Undefined init mode"
        
    def forward(self, unList):
        return self.fwdAutograd(unList, self.table)
