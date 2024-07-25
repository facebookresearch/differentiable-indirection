# Efficient Graphics Representation with Differentiable Indirection
### <i>In SIGGRAPH ASIA '23 Conference Proceedings</i>

[Webpage](https://sayan1an.github.io/din.html)

[Paper + Supplemental](https://arxiv.org/abs/2309.08387)

# Important files

* `networksBase.py` -- Defines differentiable arrays `SpatialGrid2D, SpatialGrid3D, and SpatialGrid4D`.
* `disneyFit/networks.py` -- Defines <i>Disney BRDF</i> approximation network.
* `imageCompression/networks.py` -- Defines image compression networks with varying `2D, 3D, 4D` cascaded arrays.

# A simple <i>differentiable indirection</i> example
```py
import networksBase as nb
import torch

class DifferentiableIndirection(torch.nn.Module):
    def __init__(self, primarySize, cascadedSize, torchDevice):
        super(DifferentiableIndirection, self).__init__()

        # initialize primary - gpu device, array resolutions, channel count, bilinear interpolation,
        # normalize o/p with non-linearity, scale initial content, initialize with uniform ramp - 'U'.        
        self.primary = nb.SpatialGrid2D(torchDevice, uDim=primarySize, vDim=primarySize,
                                      latent=2, bilinear=True, normalize=True, initScale=1, initMode="U")

        # initialize cascaded - gpu device, array resolutions, channel count, bilinear interpolation,
        # no o/p with non-linearity, scale initial content, initialize with constant value. 
        self.cascaded = nb.SpatialGrid2D(torchDevice, uDim=cascadedSize, vDim=cascadedSize,
                                      latent=1, bilinear=True, normalize=False, initScale=0.5, initMode="C")

    # Assumes x in [0, 1)
    def forward(self, x):
        return self.cascaded(self.primary(x))
```

# Dependencies

* `torch`
* `numpy`
* `PIL`

# Training and evaluation

Clone the repository and then follow one of the instruction sets below.

<b>Training <i>Disney BRDF</i> using cascaded-decoders of size 16.</b>
```sh
cd disneyFit
../disneyFit>python .\train.py 16 16 16
```

<b>Evaluating <i>Disney BRDF</i> with a trained network.</b>
_Note: Requires a pre-computed gbuffer to be saved in `disneyFit/in/gbuff-eval.npz`_
```sh
cd disneyFit
../disneyFit>python .\eval.py 16 16 16
```

<b>Training a 6x or 12x compressed image representation using `2D, 3D, and 4D` cascaded-array netwrok-configs.</b>
First copy the square image for compression to `imageCompression/in/MyImage/`
```sh
cd imageCompression
python ./train.py 6 "Network_p2_c2_41" "MyImage" -- 6x Compression, 2D Primary/2D Cascaded
python ./train.py 6 "Network_p2_c3_321" "MyImage" -- 6x Compression, 2D Primary/3D Cascaded
python ./train.py 6 "Network_p2_c4_401" "MyImage" -- 6x Compression, 2D Primary/4D Cascaded
python ./train.py 12 "Network_p2_c2_41" "MyImage" -- 12x Compression, 2D Primary/2D Cascaded
python ./train.py 12 "Network_p2_c3_321" "MyImage" -- 12x Compression, 2D Primary/3D Cascaded
python ./train.py 12 "Network_p2_c4_401" "MyImage" -- 12x Compression, 2D Primary/4D Cascaded
```

<b>De-compressing a pre-trained image.</b>
```sh
cd imageCompression
python ./eval.py 6 "Network_p2_c4_401" "MyImage"
```

# Contributing
See [Code of Conduct](CODE_OF_CONDUCT.md) and our [Contributing Guide](CONTRIBUTING.md).

---

Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
