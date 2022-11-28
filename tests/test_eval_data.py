import sys
sys.path.insert(0,'..')
from eval import sr_eval_ds
from torch.utils.data import DataLoader
import torchvision.transforms as T
from omegaconf import OmegaConf
import PIL

config = OmegaConf.load('../config/upsample256.yaml')


val_HR_transforms = T.Compose([
    T.Resize(256, interpolation=PIL.Image.BILINEAR),
    T.CenterCrop(256),
    T.ToTensor(),
])
val_LR_transforms = T.Compose([
    T.Resize(64, interpolation=PIL.Image.BILINEAR),
    T.CenterCrop(64),
    T.ToTensor(),
])
dataset = sr_eval_ds(config.DATA, val_HR_transforms, val_LR_transforms)
data_loader = DataLoader(
    dataset, 
    batch_size=2,
    num_workers=1,
    pin_memory=True,
    drop_last=True,
    prefetch_factor=1)

for i, (img, LR_img, embeddings, meta_info) in enumerate(data_loader):
    for k in range(2):
        save_img = T.ToPILImage()(img[k].clamp(0, 1))
        save_img.save(f'test{k}.png')
        save_img = T.ToPILImage()(LR_img[k].clamp(0, 1))
        save_img.save(f'test{k}_small.png')
    if i > 0:
        break