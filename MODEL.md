# Overview
This card describes the 64x64 text-to-image diffusion model and 256x256 upsample diffusion model described in this hana project.

# Datasets
`hana_idea64`(~2.2B Parameterss) was trained on a filtered version of the [LAION400M](https://laion.ai/blog/laion-400-open-dataset/) dataset. Our 256*256 upsampler model was also trained on the same dataset described above.

# Intended Use
We release these models to help advance research in generative modeling. Due to the limitations and biases of our model, we do not currently recommend it for commercial use.

Functionally, these models are intended to be able to perform the following tasks for research purposes:

* Generate images from natural language prompts
* Iteratively edit and refine images using super-resolution or inpainting
* Perform various downstream applications.

# Model Zoo
| Model  | Link | Parameter Size | Config file |
|--------|------|----------------|-------------|
|  `hana_idea64`      |  https://huggingface.co/hanacv/hana/resolve/main/base64.pt   |    2.2B            |   [hana_idea64.yaml](config/hana_idea64.yaml)       |
|  `hana_upsample256`     |   https://huggingface.co/hanacv/hana/resolve/main/upsampler256.pt   |       678M     |    [hana_upsample256.yaml](config/hana_upsample256.yaml)        |