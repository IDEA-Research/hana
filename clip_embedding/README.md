# Clip Embedding

### Embedding Dataset

### CC3M
```python
# CLIP Embeeding
import torch
from dataset.embedding_dataset import EmbeddingDataset


tsv_file = "/comp_robot/cv_public_dataset/ConceptualCaptions/train_resize.tsv"
bin_file = "/comp_robot/cv_public_dataset/ConceptualCaptions/CLIP_ViT_L_embedding.bin"
map_file = "/comp_robot/cv_public_dataset/ConceptualCaptions/CLIP_ViT_L_embedding.txt"
dataset = EmbeddingDataset(tsv_file, bin_file, map_file)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=512, num_workers=16)

for image_embedding, text_embedding, img, text in train_loader:
    ...
```

```python
# T5 Embeeding
import torch
from dataset.t5_embedding_dataset import T5EmbeddingDataset

tsv_file = "/comp_robot/cv_public_dataset/ConceptualCaptions/train_resize.tsv"
bin_file = "/comp_robot/cv_public_dataset/clip_embedding/t5_cc3m_train_resize.bin"
map_file = "/comp_robot/cv_public_dataset/clip_embedding/t5_cc3m_train_resize.txt"

dataset = T5EmbeddingDataset(tsv_file, bin_file, map_file)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=48, num_workers=2)

with torch.no_grad():
    for text_embedding, text, mask in train_loader:
        ...
```

### CC12M
```python
import torch
from dataset.embedding_dataset import EmbeddingDataset


tsv_file = "/comp_robot/cv_public_dataset/CC12M/tsv_all_resize/cc12m.tsv"
bin_file = "/comp_robot/cv_public_dataset/clip_embedding/CLIP_ViT_L_cc12m_embedding.bin"
map_file = "/comp_robot/cv_public_dataset/clip_embedding/CLIP_ViT_L_cc12m_embedding.txt"
dataset = EmbeddingDataset(tsv_file, bin_file, map_file)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=512, num_workers=16)

for image_embedding, text_embedding, img, text in train_loader:
    ...
```

### CombineDataset

```python
# CC3M
t5_bin_file = "/comp_robot/cv_public_dataset/clip_embedding/t5_cc3m_train_resize.bin"
t5_map_file = "/comp_robot/cv_public_dataset/clip_embedding/t5_cc3m_train_resize.txt"
clip_bin_file = "/comp_robot/cv_public_dataset/ConceptualCaptions/CLIP_ViT_L_embedding.bin"
clip_map_file = "/comp_robot/cv_public_dataset/ConceptualCaptions/CLIP_ViT_L_embedding.txt"
tsv_file = "/comp_robot/cv_public_dataset/ConceptualCaptions/train_resize.tsv"

# CC12M
t5_bin_file = "/comp_robot/cv_public_dataset/clip_embedding/t5_cc12m_train_resize.bin"
t5_map_file = "/comp_robot/cv_public_dataset/clip_embedding/t5_cc12m_train_resize.txt"
clip_bin_file = "/comp_robot/cv_public_dataset/clip_embedding/CLIP_ViT_L_cc12m_embedding.bin"
clip_map_file = "/comp_robot/cv_public_dataset/clip_embedding/CLIP_ViT_L_cc12m_embedding.txt"
tsv_file = "/comp_robot/cv_public_dataset/CC12M/tsv_all_resize/cc12m.tsv"

dataset = CLIPT5EmbeddingDataset(tsv_file, t5_bin_file, t5_map_file, clip_bin_file, clip_map_file)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=48, num_workers=2)

for clip_image_embedding, clip_text_embedding, img, clip_text_token, t5_text_embedding, t5_text_token, t5_text_mask in train_loader:
    ...
```


### Laion Embedding Dataset

```python
python laion400m_embedding.py
```