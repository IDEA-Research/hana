# Resource Allocation
```
salloc -N 4 -J idea_art  --gres=gpu:8 --cpus-per-gpu=30 --qos=ai4cvr-1 --time-min=144000  --mem 1800GB
```

# Training
## 1. text2image(base) training 
Example: ```bash train_base64.sh```
## 2. upsampler training
Example: ```bash train_up256.sh```