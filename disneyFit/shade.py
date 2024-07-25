# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from networks import *
import brdf as bdf
import utility as ut

def shadeRef(toh, noh, tol, nol, tov, nov, loh, rough, aniso, subsurf, ccGloss, metal, spec, specTint, sheen, sheenTint, albedo, ccStrength):
    dTerm = bdf.D_GGX_Disney_Aniso(rough, aniso, noh, toh) / (4 * nov)
    dTermCC = bdf.D_GGX_Disney_Clear(ccGloss, noh) / (4 * nov)
    vTerm = bdf.V_Disney(rough, aniso, nol, tol, nov, tov)
    vTermCC = bdf.V_Disney_Clear(nol, tol, nov, tov)
    diffuse = bdf.D_Disney_Diffuse(nov, nol, loh, rough, subsurf)

    luminance = 0.2126 * albedo[:, 0] + 0.7152 * albedo[:, 1] + 0.0722 * albedo[:, 2]
    luminance[luminance < 1e-5] = 1
    albedoTint = albedo / luminance[:, None]

    fresnel = bdf.D_GGX_Disney_Metal_Fresnel(loh, albedo, albedoTint, metal, spec, specTint)
    fresnelCC = bdf.D_GGX_Disney_Clear_Fresnel(loh)
    fresnelSheen = bdf.D_GGX_Disney_Sheen_Fresnel(loh, albedoTint, sheenTint)

    dvMet = dTerm * vTerm

    metTerm = fresnel * dvMet[:, None]
    diffTerm = albedo * (diffuse * (1- metal))[:, None]
    clearCoatTerm = 0.25 * ccStrength * dTermCC * vTermCC * fresnelCC
    sheenTerm = fresnelSheen * ((1 - metal) * sheen * nol)[:, None]

    baseTerm = diffTerm + metTerm + sheenTerm + clearCoatTerm[:, None]

    loh5 = torch.pow(1 - loh, 5)
   
    dvClear = dTermCC * vTermCC
    
    # compute six coefs
    cd = (1 - metal) * luminance
    cm0 = spec * 0.1 * (1 - metal) * specTint + metal * luminance
    cm1 = spec * 0.1 * (1 - metal) * (1 - specTint)
    cs0 = sheen * (1 - metal) * sheenTint
    cs1 = sheen * (1 - metal) * (1 - sheenTint)
    cc = 0.25 * ccStrength

    diffTermRecon = albedoTint * (cd * diffuse)[:,None]
    metTermRecon = albedoTint * (cm0 * (1 - loh5) * dvMet)[:, None] + (((1 - loh5) * cm1 + loh5) * dvMet)[:, None]
    sheenTermRecon = albedoTint * (cs0 * loh5 * nol)[:, None] + (cs1 * loh5 * nol)[:, None]
    clearCoatTermRecon = cc * fresnelCC * dvClear
    baseTermRecon = diffTermRecon + metTermRecon + sheenTermRecon + clearCoatTermRecon[:,None]
    
    fresnelVCol = (cm0 * (1 - loh5)) * vTerm
    fresnelVMon = ((1 - loh5) * cm1 + loh5) * vTerm

    fresnelSheen = loh5 * nol
    fresnelClear = fresnelCC * vTermCC
        
    return baseTermRecon, dTerm[:, None], dTermCC[:, None], fresnelVCol[:, None], fresnelVMon[:, None], diffuse[:, None], fresnelSheen[:, None], fresnelClear[:, None], diffuse[:, None], vTerm[:,None], vTermCC[:, None], 

def shadeNet(albedoTint, x, y, dTerm, dTermCC, fresnelVTermCol, fresnelVTermMon, diffuse, fresnelSheenCol, fresnelClear):
    ret = albedoTint * x + y
    return ret, dTerm, dTermCC, fresnelVTermCol, fresnelVTermMon, diffuse, fresnelSheenCol, fresnelClear

def shade(albedo, tangent, normal, worldPosition, metallicity, roughness, aniso, subsurf, ccGloss, ccStrength, spec, specTint, sheen, sheenTint, camPos, emitterProp, emitterHt, retIdx, quantize):
    v = ut.normNd(camPos - worldPosition)
    nov = ut.dotNd(normal, v)
    tov = ut.dotNd(tangent, v)

    outRef = torch.zeros(albedo.shape, dtype=torch.float32, device=albedo.device)
    outNet = torch.zeros(albedo.shape, dtype=torch.float32, device=albedo.device)
    
    subCg = torch.clamp(torch.stack((subsurf, ccGloss), dim=1), 0, 1-1e-6)
    spst = torch.clamp(torch.stack((spec, specTint), dim=1), 0, 1-1e-6)
    shst = torch.clamp(torch.stack((sheen, sheenTint), dim=1), 0, 1-1e-6)

    luminance = 0.2126 * albedo[:, 0] + 0.7152 * albedo[:, 1] + 0.0722 * albedo[:, 2]
    luminance[luminance < 1e-5] = 1
    albedoTint = albedo / luminance[:, None]
    lumCs = torch.clamp(torch.stack((luminance, ccStrength), dim=1), 0, 1-1e-6)

    for emIdx in range(emitterProp.shape[0]):
        emCol = emitterProp[emIdx, 0, 0:3]
        emPower = emitterProp[emIdx, 0, 3:4]
        emPos = emitterProp[emIdx, 1, 0:3]

        l = emPos - worldPosition
        emDist = ut.lengthNd(l)[:, None]
        l = l / emDist

        falloff = 5 / (0.1 * emDist * emDist + 0.8 * emDist + 0.1); 

        h = ut.normNd(l + v)

        roughNoh = torch.clamp(torch.stack((roughness, ut.dotNd(normal, h)), dim=1), 0, 1-1e-6)
        anisoToh = torch.clamp(torch.stack((aniso, ut.dotNd(tangent, h)), dim=1), 0, 1-1e-6)
        metalLoh = torch.clamp(torch.stack((metallicity, ut.dotNd(l, h)), dim=1), 0, 1-1e-6)
        nDotLV = torch.clamp(torch.stack((ut.dotNd(normal, l), nov), dim=1), 1e-3, 1-1e-6)
        tDotLV = torch.clamp(torch.stack((ut.dotNd(tangent, l), tov), dim=1), 0, 1-1e-6)
       
        if True:
            tgOut, dTermT, dTermCCT, frvCT, frvMT, diffuseHatT, frsCT, fcT, diffuseT, vTermT, vTermCCT  = shadeRef(anisoToh[:, 1], roughNoh[:,1], tDotLV[:, 0], nDotLV[:, 0], tDotLV[:, 1], nDotLV[:, 1], metalLoh[:, 1], roughNoh[:,0], anisoToh[:, 0], subCg[:, 0], subCg[:, 1], metalLoh[:, 0], spst[:, 0], spst[:, 1], shst[:, 0], shst[:, 1], albedo, lumCs[:, 1])
            x, y, dTerm, dTermCC, frvC, frvM, diffuseHat, frsC, fc = emitterHt(roughNoh, anisoToh, nDotLV, tDotLV, metalLoh, subCg, spst, shst, lumCs, vTermT, vTermCCT, diffuseT)[:9]
            if quantize == False:
                xInfer, yInfer = emitterHt.infer(roughNoh, anisoToh, nDotLV, tDotLV, metalLoh, subCg, spst, shst, lumCs)
                xInferErr = torch.mean(torch.abs(x - xInfer)).cpu().numpy()
                yInferErr = torch.mean(torch.abs(y - yInfer)).cpu().numpy()
                #print(torch.mean(torch.abs(xInfer - x)))
                assert xInferErr < 5e-8, "Inference code is not up-to-date"
                assert yInferErr < 5e-8, "Inference code is not up-to-date"
            else:
                xInfer, yInfer = emitterHt.inferQ(roughNoh, anisoToh, nDotLV, tDotLV, metalLoh, subCg, spst, shst, lumCs)
                                   
            netVal = shadeNet(albedoTint, xInfer, yInfer, dTerm, dTermCC, frvC, frvM, diffuseHat, frsC, fc)[retIdx]
            multiplier = 1
            if retIdx == 0:
                multiplier = emCol * emPower * falloff
            outRef += multiplier * (tgOut, dTermT, dTermCCT, frvCT, frvMT, diffuseHatT, frsCT, fcT)[retIdx]
            outNet += multiplier * netVal
            
        #outNet += shadeNet(albedo, emitterHt(roughNoh, metalVoh, nDotLV)[0])
    return outRef, outNet

def render(torchDevice, emitterHt, gBufData, retIdx = 0, quantize=False):
    ssAlbedoCpu = gBufData["ssAlbedo"][:,:,0:3]
    ssMatIdCpu = np.around(gBufData["ssAlbedo"][:,:,3] * 255).astype(int)
    ssTangentCpu = gBufData["ssTangent"][:,:,0:3]
    ssNormalCpu = gBufData["ssNormal"][:,:,0:3]
    ssMetalCpu = gBufData["ssNormal"][:,:,3]
    
    ssWorldPosCpu =  gBufData["ssWorldPos"][:,:,0:3]
    ssRoughCpu = gBufData["ssWorldPos"][:,:,3]
    ssSubsurfaceCpu = np.clip(gBufData["ssSubsurface"][:,:,0], 0, 1-1e-6)
    ssAnisoCpu = gBufData["ssSubsurface"][:,:,1]
    ssCcSCpu = gBufData["ssUv"][:,:,2] # clear coat strength
    ssCcGCpu = gBufData["ssUv"][:,:,3] # clear coat gloss
    ssSpecCpu = gBufData["ssSpecSheen"][:,:,0]
    ssSpecTintCpu = gBufData["ssSpecSheen"][:,:,1]
    ssSheenCpu = gBufData["ssSpecSheen"][:,:,2]
    ssSheenTintCpu = gBufData["ssSpecSheen"][:,:,3]

    ssUvCpu = np.clip(gBufData["ssUv"][:,:,0:2], 0, 1-1e-6)

    # test = True

    # if test:
    #     ssAlbedoCpu[ssMatIdCpu == 8, :] = np.array([0.75, 0.83, 0.46]) * 0.7
    #     ssMetalCpu[ssMatIdCpu == 8] = 0.2
    #     ssRoughCpu[ssMatIdCpu == 8] = 0.8
    #     ssAnisoCpu[ssMatIdCpu == 8] = 0
    #     ssSpecCpu[ssMatIdCpu == 8] = 0.0
    #     ssSpecTintCpu[ssMatIdCpu == 8] = 0.0
    #     ssSubsurfaceCpu[ssMatIdCpu == 8] = 0.05
    #     ssSheenCpu[ssMatIdCpu == 8] = 0.0
    #     ssSheenTintCpu[ssMatIdCpu == 8] = 0.0       
    
    renderHeight = ssNormalCpu.shape[0]
    renderWidth = ssNormalCpu.shape[1]

    renderTileFrac = 8
    tileWidth = int(np.around(renderWidth / renderTileFrac))
    tileHeight = int(np.around(renderHeight / renderTileFrac))
    outTexNet = np.zeros((renderHeight, renderWidth, 3), dtype=np.float32)
    outTexRef = np.zeros((renderHeight, renderWidth, 3), dtype=np.float32)
    
    camPos = torch.from_numpy(gBufData["camPos"].reshape((1,3))).to(torchDevice)
    emitterProp = torch.from_numpy(gBufData["emitterProp"]).to(torchDevice)

    for h in range(renderTileFrac):
        hOffset = h * tileHeight
        for w in range(renderTileFrac):
            wOffset = w * tileWidth
            
            albedo = torch.from_numpy(ssAlbedoCpu[hOffset : hOffset + tileHeight, wOffset : wOffset + tileWidth]).to(torchDevice).reshape((-1, 3))
            tangent = torch.from_numpy(ssTangentCpu[hOffset : hOffset + tileHeight, wOffset : wOffset + tileWidth]).to(torchDevice).reshape((-1, 3))
            normal = torch.from_numpy(ssNormalCpu[hOffset : hOffset + tileHeight, wOffset : wOffset + tileWidth]).to(torchDevice).reshape((-1, 3))
            metal = torch.from_numpy(ssMetalCpu[hOffset : hOffset + tileHeight, wOffset : wOffset + tileWidth]).to(torchDevice).reshape((-1))
            worldPos = torch.from_numpy(ssWorldPosCpu[hOffset : hOffset + tileHeight, wOffset : wOffset + tileWidth]).to(torchDevice).reshape((-1, 3))
            rough = torch.from_numpy(ssRoughCpu[hOffset : hOffset + tileHeight, wOffset : wOffset + tileWidth]).to(torchDevice).reshape((-1))
            ccGloss = torch.from_numpy(ssCcGCpu[hOffset : hOffset + tileHeight, wOffset : wOffset + tileWidth]).to(torchDevice).reshape((-1))
            ccStrength = torch.from_numpy(ssCcSCpu[hOffset : hOffset + tileHeight, wOffset : wOffset + tileWidth]).to(torchDevice).reshape((-1))
            aniso = torch.from_numpy(ssAnisoCpu[hOffset : hOffset + tileHeight, wOffset : wOffset + tileWidth]).to(torchDevice).reshape((-1))
            subsurf = torch.from_numpy(ssSubsurfaceCpu[hOffset : hOffset + tileHeight, wOffset : wOffset + tileWidth]).to(torchDevice).reshape((-1))
            spec = torch.from_numpy(ssSpecCpu[hOffset : hOffset + tileHeight, wOffset : wOffset + tileWidth]).to(torchDevice).reshape((-1))
            specTint = torch.from_numpy(ssSpecTintCpu[hOffset : hOffset + tileHeight, wOffset : wOffset + tileWidth]).to(torchDevice).reshape((-1))
            sheen = torch.from_numpy(ssSheenCpu[hOffset : hOffset + tileHeight, wOffset : wOffset + tileWidth]).to(torchDevice).reshape((-1))
            sheenTint = torch.from_numpy(ssSheenTintCpu[hOffset : hOffset + tileHeight, wOffset : wOffset + tileWidth]).to(torchDevice).reshape((-1))
            
            oRef, oNet = shade(albedo, tangent, normal, worldPos, metal, rough, aniso, subsurf, ccGloss, ccStrength, spec, specTint, sheen, sheenTint, camPos, emitterProp, emitterHt, retIdx, quantize)
            outTexRef[hOffset : hOffset + tileHeight, wOffset : wOffset + tileWidth] = oRef.cpu().numpy().reshape((tileHeight, tileWidth, -1))
            outTexNet[hOffset : hOffset + tileHeight, wOffset : wOffset + tileWidth] = oNet.cpu().numpy().reshape((tileHeight, tileWidth, -1))

    return outTexNet, outTexRef

# op0, op1 = render(0, ut.getTorchDevice("cuda"), None, 3)
# plt.imshow(op1)
# plt.show()