# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import utility as ut

ROUGHNESS_MIN = 0.01

def D_GGX(roughness, nDotH):
    a2 = roughness**2
    f2 = (1.0 + (a2 - 1) * (nDotH**2))**2
    return a2 / (torch.clamp(f2 * np.pi, min=ut.FLOAT_MIN))

def D_GGX_Disney(roughness, nDotH):
    a2 = torch.clamp(roughness * roughness, min=ROUGHNESS_MIN * ROUGHNESS_MIN)
    p = (1 + (a2 * a2 - 1) * nDotH**2) / a2

    return 1 / (np.pi * p * p)

def D_GGX_Disney_Clear(cgl, nDotH):
    ag = (1 - cgl) * 0.1 + cgl * 0.001
    ag2 = ag**2

    ag2m1 = ag2 - 1
    nDotH2 = nDotH**2

    return ag2m1 / (np.pi * torch.log(ag2) * (1 + ag2m1 * nDotH2))

def F_D(f, nDotU):
    return 1 + (f - 1) * torch.pow((1 - nDotU), 5)

def F_BaseDiff(nDotV, nDotL, vDotH, roughness):
    fd90 = 0.5 + 2 * roughness * (vDotH**2)
    return F_D(fd90, nDotV) * F_D(fd90, nDotL) * nDotL / np.pi

def F_Subsurface(nDotV, nDotL, vDotH, roughness):
    fs90 = roughness * (vDotH**2)
    return 1.25 * (F_D(fs90, nDotV) * F_D(fs90, nDotL) * (1 / (nDotV + nDotL) - 0.5) + 0.5) * nDotL / np.pi

def D_Disney_Diffuse(nDotV, nDotL, vDotH, roughness, subsurface):
    return (1 - subsurface) * F_BaseDiff(nDotV, nDotL, vDotH, roughness) + subsurface * F_Subsurface(nDotV, nDotL, vDotH, roughness)

def anisoGgxPrecom(roughness, aniso):
    aspect = torch.sqrt(1 - 0.9 * aniso)

    ax = torch.clamp(roughness * roughness / aspect, min=ROUGHNESS_MIN * ROUGHNESS_MIN)
    ay = torch.clamp(roughness * roughness * aspect, min=ROUGHNESS_MIN * ROUGHNESS_MIN)
    
    ax2 = ax**2
    ay2 = ay**2

    ct = torch.sqrt(np.pi / (ax * ay)**3)

    return (ay2 - ax2) * ct, (ay2 - 1) * ax2 * ct, ax2 * ct

def anisoGgxPrecom2(roughness, aniso):
    aspect = torch.sqrt(1 - 0.9 * aniso)

    ax = torch.clamp(roughness * roughness / aspect, min=ROUGHNESS_MIN * ROUGHNESS_MIN)
    ay = torch.clamp(roughness * roughness * aspect, min=ROUGHNESS_MIN * ROUGHNESS_MIN)
    
    ax2 = ax**2
    ay2 = ay**2

    ctInv = torch.sqrt((ax * ay)**3 / np.pi)

    return (ax2 - ay2), (1 - ay2) * ax2, ax2, ctInv**2

def clearPrecomN(nov, cgl):
    ag = (1 - cgl) * 0.1 + cgl * 0.001
    ag2 = ag**2
    ag2m1 = ag2 - 1
    
    novInv = 1 / (4 * nov)

    c0 = 4 * nov * np.pi * torch.log(ag2) 
    c1 = 4 * nov * np.pi * torch.log(ag2) / ag2m1
    
    return c0, c1, novInv

def clearPrecom(cgl):
    ag = (1 - cgl) * 0.1 + cgl * 0.001
    ag2 = ag**2
    ag2m1 = ag2 - 1
    
    c0 = np.pi * torch.log(ag2) 
    c1 = np.pi * torch.log(ag2) / ag2m1
    
    return c0, c1

def clearPrecom2(cgl):
    ag = (1 - cgl) * 0.1 + cgl * 0.001
    ag2 = ag**2
    ag2m1 = 1 - ag2
    
    c0 = ag2m1
    c1 = -np.pi * torch.log(ag2)
        
    return c0, c1

def D_GGX_Disney_Aniso(roughness, aniso, nDotH, tDotH):
    aspect = torch.sqrt(1 - 0.9 * aniso)

    ax = torch.clamp(roughness * roughness / aspect, min=ROUGHNESS_MIN * ROUGHNESS_MIN)
    ay = torch.clamp(roughness * roughness * aspect, min=ROUGHNESS_MIN * ROUGHNESS_MIN)
    
    ax2 = ax**2
    ay2 = ay**2
    tDotH2 = tDotH**2
    nDotH2 = nDotH**2
    
    l = (ay2 - ax2) * tDotH2 + ax2 * (1 - nDotH2 * (1 - ay2))

    ax3 = ax2 * ax
    ay3 = ay2 * ay
    
    return ax3 * ay3 / (np.pi * l * l)

def D_GGX_Disney_Metal_Fresnel(lDotH, albedo, albedoTint, metal, spec, specTint):
    ks = (1 - specTint)[:, None] + specTint[:, None] * albedoTint
    c0 = (spec * 0.1 * (1 - metal))[:, None] * ks + metal[:, None] * albedo

    return c0 + (1 - c0) * torch.pow(1 - lDotH, 5)[:, None]

def D_GGX_Disney_Clear_Fresnel(vDotH):
    return 0.04 + 0.96 * torch.pow(1 - vDotH, 5)

def D_GGX_Disney_Sheen_Fresnel(lDotH, albedoTint, sheenTint):
    cs = (1 - sheenTint)[:, None] + sheenTint[:, None] * albedoTint
    return cs * torch.pow(1 - lDotH, 5)[:, None]

def D_GGX_Base(roughness, nDotH):
    a2 = roughness * roughness
    f = (a2 * nDotH - nDotH) * nDotH + 1.0
    return a2 / (np.pi * f * f)

def D_GGX_Original(roughness, nDotH):
    a2 = a * a
    cos2 = nDotH * nDotH
    a2tan2 = a2 + (1 - cos2) / cos2

    return a2 / (np.pi * cos2 * cos2 * a2tan2 * a2tan2)

def D_GGX_Original_Simplified(roughness, nDotH):
    a2 = a * a
    cos2 = nDotH * nDotH
    a2tan2cos2 = a2 * cos2 + (1 - cos2)

    return a2 / (np.pi * a2tan2cos2 * a2tan2cos2)

def G1_Disney(ax, ay, wz, wx):
    wx2 = wx**2
    wz2 = wz**2
    wy2 = 1 - wx2 - wz2
    ax2 = ax**2
    ay2 = ay**2

    d = (torch.sqrt(1 + (ax2 * wx2 + ay2 * wy2) / wz2) - 1) * 0.5 

    return 1 / (1 + d)

def V_Disney(roughness, aniso, nDotL, tDotL, nDotV, tDotV):
    aspect = torch.sqrt(1 - 0.9 * aniso)

    ax = torch.clamp(roughness * roughness / aspect, min=ROUGHNESS_MIN * ROUGHNESS_MIN)
    ay = torch.clamp(roughness * roughness * aspect, min=ROUGHNESS_MIN * ROUGHNESS_MIN)
    
    return G1_Disney(ax, ay, nDotL, tDotL) * G1_Disney(ax, ay, nDotV, tDotV)

def V_Disney_Clear(nDotL, tDotL, nDotV, tDotV):
    return G1_Disney(0.25, 0.25, nDotL, tDotL) * G1_Disney(0.25, 0.25, nDotV, tDotV)


def G1_GGX(NoU, roughness):
    cos2 = NoU * NoU
    tan2 = (1 - cos2) / cos2
    a2 = roughness * roughness

    return 2 / (1 + torch.sqrt(1 + a2 * tan2))

def V_SmithGGX_Original(NoV, NoL, roughness):
    return G1_GGX(NoV, roughness) * G1_GGX(NoL, roughness)

def F_Schlick(u, f0):
    return f0 + (1 - f0) * ((1.0 - u)**5)[:,None]

def rougnessLogSample(batchSize):
    a = np.random.uniform(size=(batchSize//4), low=ROUGHNESS_MIN, high=2*ROUGHNESS_MIN).astype(np.float32)
    b = np.random.uniform(size=(batchSize//4), low=2*ROUGHNESS_MIN, high=5*ROUGHNESS_MIN).astype(np.float32)
    c = np.random.uniform(size=(batchSize//4), low=5*ROUGHNESS_MIN, high=20*ROUGHNESS_MIN).astype(np.float32)
    d = np.random.uniform(size=(batchSize - 3 * batchSize//4), low=20*ROUGHNESS_MIN, high=1-1e-6).astype(np.float32)

    s = np.hstack((c,d,a,b))
    np.random.shuffle(s)
    return torch.from_numpy(s)
