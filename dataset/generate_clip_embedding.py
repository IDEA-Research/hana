# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------------------------
# Generate CLIP Embedding
# ------------------------------------------------------------------------------------------------

import os
import json
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import clip


class ToyDataset(Dataset):
    def __init__(self, json_file, transforms=None, **kwargs) -> None:
        super().__init__()
        with open(json_file, 'r') as f:
            mapping = json.load(f)
        # fetch all indexes
        self.indexes = sorted(list(mapping.keys()))
        self.mapping = mapping
        self.transforms = transforms
    
    def __len__(self) -> int:
        return len(self.indexes)
    
    def __getitem__(self, index):
        total_index = self.indexes[index]
        img_path = self.mapping[total_index]['img_path']
        caption = self.mapping[total_index]['caption']
        caption = clip.tokenize(caption, truncate=True).squeeze()
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, caption, total_index


def run_embedding(json_file, outdir, batch_size=512):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load('ViT-L/14', device=device)
    
    assert os.path.exists(json_file), 'Mapping file does not exist: {}'.format(json_file)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    dataset = ToyDataset(json_file, transforms=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=12)
    
    with torch.no_grad():
        for image, text, index in tqdm(dataloader):
            image = image.to(device)
            text = text.to(device)
            
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            for i, idx in enumerate(index):
                img_feat = image_features[i].cpu().numpy()
                txt_feat = text_features[i].cpu().numpy()
                img_emb_path = os.path.join(outdir, '{}.img_emb'.format(idx))
                txt_emb_path = os.path.join(outdir, '{}.txt_emb'.format(idx))
                with open(img_emb_path, 'wb') as img_fb, open(txt_emb_path, 'wb') as txt_fb:
                    img_fb.write(img_feat.tobytes())
                    txt_fb.write(txt_feat.tobytes())

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    args = parser.parse_args()
    run_embedding(args.json_file, args.outdir, args.batch_size)