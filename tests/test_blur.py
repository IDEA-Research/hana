from PIL import Image
import numpy as np
import albumentations
from kornia.filters import gaussian_blur2d
import torchvision.transforms as T

img_path = '/comp_robot/mm_generative/eval/prompts/training_eval/sr_images/1_anya.jpeg'
img = Image.open(img_path)
img.save('original.png')

sigmas = [0.6, 10]
for sigma in sigmas:
    img = np.array(img).astype(np.uint8)
    LR_img = albumentations.GaussianBlur(
        blur_limit=(5,5), sigma_limit=(sigma, sigma), p=1.
    )(image=img)['image']    

    blured = Image.fromarray(LR_img)
    blured.save(f'{sigma}.png')

    tensor = T.ToTensor()(img)[None, ...]
    blured_2 = gaussian_blur2d(tensor, (5, 5), (sigma, sigma))
    transform = T.ToPILImage()
    blured2 = transform(blured_2[0])
    blured2.save(f'{sigma}_2.png')