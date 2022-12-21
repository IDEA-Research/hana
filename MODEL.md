# Overview
This card describes the 64x64 text-to-image diffusion model and 256x256 upsample diffusion model described in this hana project.

# Datasets
`hana_idea64`(~2.2B Parameters) and `hana_upsample256`(~678M Parameters) were trained on a filtered version of the [LAION400M](https://laion.ai/blog/laion-400-open-dataset/) dataset.

# Intended Use
We release pretrained models to help advance research in generative modeling. Due to the limitations and biases inherited from training data, we do not recommend it for commercial use.

Functionally, the released models are intended for the following tasks for research purposes:

* Generate images given natural language prompts (text-to-image).
* Iteratively edit and refine images using super-resolution or inpainting.
* Enable downstream applications such as editing, personalization and text-to-3D.

# Model Zoo
|Model   |Link  |Parameter Size  |Config file |
|--------|------|----------------|-----------|
|  `hana_idea64`      |  https://huggingface.co/hanacv/hana/resolve/main/base64.pt   |    2.2B            |  [hana_idea64.yaml](config/hana_idea64.yaml)       |
|  `hana_upsample256`     |   https://huggingface.co/hanacv/hana/resolve/main/upsampler256.pt   |       678M     |   [hana_upsample256.yaml](config/hana_upsample256.yaml)        |
