# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import brdf as bdf
import sys
sys.path.insert(1, '../')
#from shade import *
from networksBase import *
import utility as ut

def quantize(input, bits):
    bitMax = (2 ** bits) - 1
    q = torch.round(torch.clamp(input, 0, 1) * bitMax).to(torch.long)
    return torch.clamp(q.to(torch.float) / bitMax, 0, 1-1e-6)

class DisneyNet(nn.Module):
    def __init__(self, torchDevice, resHDR0, resSDR0, resSDR1):
        super(DisneyNet, self).__init__()
        
        baseRoughAnisoRes = 2048
        self.tabRA = torch.nn.Parameter(torch.ones((baseRoughAnisoRes, baseRoughAnisoRes, 3), dtype=torch.float, requires_grad=False, device=torchDevice))
        self.tabRA.requires_grad_(False)
        self.tabCg = torch.nn.Parameter(torch.ones((baseRoughAnisoRes, 2), dtype=torch.float, requires_grad=False, device=torchDevice))
        self.tabCg.requires_grad_(False)
     
        uvIn = ut.generateUvCoord(torchDevice, baseRoughAnisoRes, baseRoughAnisoRes).reshape((-1, 2))
        
        c0, c1, c2 = bdf.anisoGgxPrecom(uvIn[:, 0], uvIn[:, 1])
        c0 = c0.reshape((baseRoughAnisoRes, baseRoughAnisoRes))
        c1 = c1.reshape((baseRoughAnisoRes, baseRoughAnisoRes))
        c2 = c2.reshape((baseRoughAnisoRes, baseRoughAnisoRes))
        self.tabRA[:,:, 0] = torch.flip(c0, [0])
        self.tabRA[:,:, 1] = torch.flip(c1, [0])
        self.tabRA[:,:, 2] = torch.flip(c2, [0])

        c0, c1 = bdf.clearPrecom(ut.generateUCoord(torchDevice, baseRoughAnisoRes))
        self.tabCg[:, 0] = c0
        self.tabCg[:, 1] = c1

        self.tabRAEncoder = SpatialGrid2D(torchDevice, uDim=baseRoughAnisoRes, vDim=baseRoughAnisoRes, latent=2, bilinear=False, normalize=True, initScale=1, initMode="U")
        self.tabRADecoder = SpatialGrid2D(torchDevice, uDim=resHDR0, vDim=resHDR0, latent=3, bilinear=True, normalize=False, initScale=1, initMode="R")

        self.tabCgEncoder = SpatialGrid1D(torchDevice, uDim=baseRoughAnisoRes, latent=1, bilinear=False, normalize=True, initScale=1, initMode="U")
        self.tabCgDecoder = SpatialGrid1D(torchDevice, uDim=resHDR0*2, latent=2, bilinear=True, normalize=False, initScale=1, initMode="R")

        with torch.no_grad():
            uvIn = ut.generateUvCoord(torchDevice, resHDR0, resHDR0).reshape((-1, 2))
        
            c0, c1, c2 = bdf.anisoGgxPrecom(uvIn[:, 0], uvIn[:, 1])
            c0 = c0.reshape((resHDR0, resHDR0))
            c1 = c1.reshape((resHDR0, resHDR0))
            c2 = c2.reshape((resHDR0, resHDR0))
            self.tabRADecoder.table[:,:, 0] = torch.flip(c0, [0])
            self.tabRADecoder.table[:,:, 1] = torch.flip(c1, [0])
            self.tabRADecoder.table[:,:, 2] = torch.flip(c2, [0])

            c0, c1 = bdf.clearPrecom(ut.generateUCoord(torchDevice, resHDR0*2))
            self.tabCgDecoder.table[:, 0] = c0
            self.tabCgDecoder.table[:, 1] = c1

        self.tabRSEncoder = SpatialGrid2D(torchDevice, uDim=64, vDim=64, latent=3, bilinear=True, normalize=True, initScale=1, initMode="R")

        self.tabLVR = SpatialGrid2D(torchDevice, uDim=resSDR0, vDim=resSDR0, latent=1, bilinear=True, normalize=True, initScale=1, initMode="R")
        self.tabLV = SpatialGrid2D(torchDevice, uDim=resSDR1, vDim=resSDR1, latent=4, bilinear=True, normalize=True, initScale=1, initMode="R")
        self.tabNHL = SpatialGrid2D(torchDevice, uDim=resSDR1, vDim=resSDR1, latent=4, bilinear=True, normalize=True, initScale=1, initMode="R")

    def parameters(self):
        return list(self.tabRSEncoder.parameters()) + list(self.tabRAEncoder.parameters()) + list(self.tabRADecoder.parameters()) + list(self.tabCgEncoder.parameters()) + list(self.tabCgDecoder.parameters()) + list(self.tabLVR.parameters()) + list(self.tabLV.parameters()) + list(self.tabNHL.parameters())

    def state_dict(self):
        od = OrderedDict()
        
        od['tablvr'] = self.tabLVR.state_dict()
        od['tablv'] = self.tabLV.state_dict()
        od['tabnhl'] = self.tabNHL.state_dict()

        od['tabRSEnc'] = self.tabRSEncoder.state_dict()

        od['tabRAEnc'] = self.tabRAEncoder.state_dict()
        od['tabRADec'] = self.tabRADecoder.state_dict()

        od['tabCgEnc'] = self.tabCgEncoder.state_dict()
        od['tabCgDec'] = self.tabCgDecoder.state_dict()

        return od

    def load_state_dict(self, stateDict):
        self.tabLVR.load_state_dict(stateDict['tablvr'])
        self.tabLV.load_state_dict(stateDict['tablv'])
        self.tabNHL.load_state_dict(stateDict['tabnhl'])

        self.tabRSEncoder.load_state_dict(stateDict['tabRSEnc'])

        self.tabRAEncoder.load_state_dict(stateDict['tabRAEnc'])
        self.tabRADecoder.load_state_dict(stateDict['tabRADec'])

        self.tabCgEncoder.load_state_dict(stateDict['tabCgEnc'])
        self.tabCgDecoder.load_state_dict(stateDict['tabCgDec'])
    
    def encoder(self, roughAniso, subCg, metallicity, lumCs, spst, shst):
        raEnc = self.tabRAEncoder(roughAniso)
        cgEnc = self.tabCgEncoder(subCg[:, 1])
        rsEnc = self.tabRSEncoder(torch.cat((roughAniso[:, 0:1] , subCg[:,0:1]), dim=1))
    
        cd = (1 - metallicity) * lumCs[:, 0]
        cm0 = spst[:, 0] * 0.1 * (1 - metallicity) * spst[:, 1] + metallicity * lumCs[:, 0]
        cm1 = spst[:, 0] * 0.1 * (1 - metallicity) * (1 - spst[:, 1])
        cs0 = shst[:, 0] * (1 - metallicity) * shst[:, 1]
        cs1 = shst[:, 0] * (1 - metallicity) * (1 - shst[:, 1])
        cc = 0.25 * 0.25 * lumCs[:, 1]

        return raEnc, cgEnc, rsEnc * cd[:, None], cm0[:, None], cm1[:, None], cs0[:, None], cs1[:, None], cc[:, None], rsEnc
    
    def precompute(self):
        with torch.no_grad():
            self.tabRADecoderQ = self.tabRADecoder.table.half()
            self.tabCgDecoderQ = self.tabCgDecoder.table.half()
            self.tabLVRQ = quantize(triangle(self.tabLVR.table), 16)
            self.tabLVQ = quantize(triangle(self.tabLV.table), 16)
            self.tabNHLQ = quantize(triangle(self.tabNHL.table), 16)

    # this is the inference only code
    def infer(self, roughNoh, anisoToh, nDotLV, tDotLV, metalLoh, subCg, spst, shst, lumCs):
        with torch.no_grad():
            # encoder part - encodes material parameters to latent space
            # transform 10D input to 11D latent vector
            # inputs are roughness, anisotropy, subsurface, clear-coat gloss, metallicity, luminance, clear-coat strength, specular, specular-tint, sheen, sheen-tint
            # in practice, the output of the encoder should be stored as material texture
            # Note no geometric terms are present in the input of the encoder
            ra, cg, rsCd, cm0, cm1, cs0, cs1, cc = self.encoder(torch.stack((roughNoh[:,0], anisoToh[:,0]), dim=1), subCg, metalLoh[:, 0], lumCs, spst, shst)[:8]
             
            # Disney evalution
            raCoefs = self.tabRADecoder(ra)
            cgCoefs = self.tabCgDecoder(cg[:, 0])
            vTerm = self.tabLVR(torch.cat((nDotLV[:, 0:1] * nDotLV[:, 1:2], ra[:,0:1]), dim=1)) 
            nlvCoefs = self.tabLV(nDotLV)
            nhlCoefs = self.tabNHL(torch.stack((metalLoh[:, 1], nDotLV[:, 0]), dim=1))

            noh2 = roughNoh[:,1]**2
            den = raCoefs[:, 0] * anisoToh[:,1]**2 + raCoefs[:, 1] * noh2 + raCoefs[:, 2]
                
            dTerm = 0.25 / (nDotLV[:,1] * den**2)[:,None]
            
            x = (rsCd[:, 0:1] * metalLoh[:,1:2] * nlvCoefs[:, 1:2] + rsCd[:, 1:2] * nlvCoefs[:, 2:3] + rsCd[:, 2:3] * nlvCoefs[:, 3:4]) + (cm0 * vTerm * nhlCoefs[:, 2:3]) * dTerm + cs0 * nhlCoefs[:, 0:1]
            y = ((1 - cm1) * nhlCoefs[:, 3:4] + cm1) * vTerm * dTerm + cs1 * nhlCoefs[:, 0:1] + cc * nhlCoefs[:, 1:2] *  nlvCoefs[:, 0:1] / (nDotLV[:,1] * (cgCoefs[:,0] * noh2 + cgCoefs[:, 1]))[:,None]

        # output = albedo * x + y
        return x, y
    
    # this is the inference only code but with quantization
    def inferQ(self, roughNoh, anisoToh, nDotLV, tDotLV, metalLoh, subCg, spst, shst, lumCs):
        with torch.no_grad():
            # encoder part - encodes material parameters to latent space
            # transform 10D input to 11D latent vector
            # inputs are roughness, anisotropy, subsurface, clear-coat gloss, metallicity, luminance, clear-coat strength, specular, specular-tint, sheen, sheen-tint
            # in practice, the output of the encoder should be stored as material texture
            # Note no geometric terms are present in the input of the encoder
            ra, cg, rsCd, cm0, cm1, cs0, cs1, cc = self.encoder(torch.stack((roughNoh[:,0], anisoToh[:,0]), dim=1), subCg, metalLoh[:, 0], lumCs, spst, shst)[:8]
            
            ra = quantize(ra, 16)
            cg = quantize(cg, 16)
            rsCd = quantize(rsCd, 16)
            cm0 = quantize(cm0, 16)
            cm1 = quantize(cm1, 16)
            cs0 = quantize(cs0, 16)
            cs1 = quantize(cs1, 16)
            cc = quantize(cc, 16)

            # Disney evalution
            raCoefs = bilinear2d(ra, self.tabRADecoderQ, False)[0]
            cgCoefs = linear1d(cg[:, 0], self.tabCgDecoderQ, False)[0]
            vTerm = bilinear2d(torch.cat((nDotLV[:, 0:1] * nDotLV[:, 1:2], ra[:,0:1]), dim=1), self.tabLVRQ, False)[0]
            nlvCoefs = bilinear2d(nDotLV, self.tabLVQ, False)[0]
            nhlCoefs = bilinear2d(torch.stack((metalLoh[:, 1], nDotLV[:, 0]), dim=1), self.tabNHLQ, False)[0]

            noh2 = roughNoh[:,1]**2
            den = raCoefs[:, 0] * anisoToh[:,1]**2 + raCoefs[:, 1] * noh2 + raCoefs[:, 2]
                
            dTerm = 0.25 / (nDotLV[:,1] * den**2)[:,None]
            
            x = (rsCd[:, 0:1] * metalLoh[:,1:2] * nlvCoefs[:, 1:2] + rsCd[:, 1:2] * nlvCoefs[:, 2:3] + rsCd[:, 2:3] * nlvCoefs[:, 3:4]) + (cm0 * vTerm * nhlCoefs[:, 2:3]) * dTerm + cs0 * nhlCoefs[:, 0:1]
            y = ((1 - cm1) * nhlCoefs[:, 3:4] + cm1) * vTerm * dTerm + cs1 * nhlCoefs[:, 0:1] + cc * nhlCoefs[:, 1:2] *  nlvCoefs[:, 0:1] / (nDotLV[:,1] * (cgCoefs[:,0] * noh2 + cgCoefs[:, 1]))[:,None]

        # output = albedo * x + y
        return x, y
    
    # this is the training code
    def forward(self, roughNoh, anisoToh, nDotLV, tDotLV, metalLoh, subCg, spst, shst, lumCs, vTermTarget, vTermCCTarget, diffuseTarget):
        # encoder part - encodes material parameters to latent space
        # transform 10D input to 11D latent vector
        # inputs are roughness, anisotropy, subsurface, clear-coat gloss, metallicity, luminance, clear-coat strength, specular, specular-tint, sheen, sheen-tint
        # in practice, the output of the encoder should be stored as material texture
        # Note no geometric terms are present in the input of the encoder
        ra, cg, rsCd, cm0, cm1, cs0, cs1, cc, rsEnc = self.encoder(torch.stack((roughNoh[:,0], anisoToh[:,0]), dim=1), subCg, metalLoh[:, 0], lumCs, spst, shst)

        raCoefs = self.tabRADecoder(ra)
        cgCoefs = self.tabCgDecoder(cg[:, 0])
        vTerm = self.tabLVR(torch.cat((nDotLV[:, 0:1] * nDotLV[:, 1:2], ra[:,0:1]), dim=1)) 
        nlvCoefs = self.tabLV(nDotLV)
        nhlCoefs = self.tabNHL(torch.stack((metalLoh[:, 1], nDotLV[:, 0]), dim=1))

        # compute some trainable intermediate quantitities
        
        # training targets
        loh5 = triangleWave(torch.pow(1 - metalLoh[:, 1:2], 5))
        fresnelCC = bdf.D_GGX_Disney_Clear_Fresnel(metalLoh[:, 1:2])
        cd = (1 - metalLoh[:, 0:1]) * lumCs[:, 0:1]
        raCoefsTarget = bilinear2d(torch.stack((roughNoh[:,0], anisoToh[:,0]), dim=1), self.tabRA)[0]
        cgCoefsTarget = linear1d(subCg[:,1], self.tabCg)[0]
        cm0ProdTarget = cm0 * (1 - loh5) * vTermTarget
        csProdTarget = loh5 * nDotLV[:, 0:1]
        cm1ProdTarget = ((1 - loh5) * cm1 + loh5) * vTermTarget
        ccProdTarget = fresnelCC * vTermCCTarget

        # compute equivalent network output
        diffuse = rsEnc[:, 0:1] * metalLoh[:,1:2] * nlvCoefs[:, 1:2] + rsEnc[:, 1:2] * nlvCoefs[:, 2:3] + rsEnc[:, 2:3] * nlvCoefs[:, 3:4]
        cm0Prod = cm0 * vTerm * nhlCoefs[:, 2:3]
        cm1Prod = ((1 - cm1) * nhlCoefs[:, 3:4] + cm1) * vTerm
        csProd = nhlCoefs[:, 0:1]
        ccProd = nhlCoefs[:, 1:2] *  nlvCoefs[:, 0:1]
        
        # compute the errors
        errRoughnessCoef = torch.mean(torch.abs((raCoefsTarget - raCoefs)))
        errClearGlossCoef = torch.mean(torch.abs((cgCoefsTarget - cgCoefs)))
        errDiffTerm = torch.mean(torch.abs(diffuseTarget - diffuse))
        errCm0Prod = torch.mean(torch.abs(cm0ProdTarget - cm0Prod))
        errCm1Prod = torch.mean(torch.abs(cm1ProdTarget - cm1Prod))
        errCsProd = torch.mean(torch.abs(csProdTarget - csProd))
        errCcProd = torch.mean(torch.abs(ccProdTarget - ccProd))

        finalErr = (errRoughnessCoef + errClearGlossCoef + errDiffTerm + errCm0Prod + errCm1Prod + errCsProd + errCcProd) / 7.0

        # some other intermediate results for visualization
        noh2 = roughNoh[:,1]**2
        den = raCoefs[:, 0] * anisoToh[:,1]**2 + raCoefs[:, 1] * noh2 + raCoefs[:, 2]
                
        dTerm = 0.25 / (nDotLV[:,1] * den**2)[:,None]
        dTermCC = 1 / (nDotLV[:,1] * (cgCoefs[:,0] * noh2 + cgCoefs[:, 1]))[:,None]
     
        with torch.no_grad():
            x = cd * diffuse + cm0Prod * dTerm + cs0 * csProd
            y = cm1Prod * dTerm + cs1 * csProd + cc *ccProd * dTermCC

        return x , y, dTerm, 0.25 * dTermCC, cm0Prod, cm1Prod, diffuse, csProd, ccProd, finalErr