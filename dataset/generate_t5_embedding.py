"""Generate T5 Embedding
"""
import os
import json

import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel


class ToyDataset(Dataset):
    def __init__(self, json_file, **kwargs) -> None:
        super().__init__()
        with open(json_file, 'r') as f:
            mapping = json.load(f)
        # fetch all indexes
        self.indexes = sorted(list(mapping.keys()))
        self.mapping = mapping
        self.tokenizer = T5Tokenizer.from_pretrained("t5-11b")
    
    def __len__(self) -> int:
        return len(self.indexes)
    
    def __getitem__(self, index):
        total_index = self.indexes[index]
        caption = self.mapping[total_index]['caption']
        encoding = self.tokenizer(
            caption,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt"
        )
        input_ids, attention_mask = encoding.input_ids.squeeze(), encoding.attention_mask.squeeze()
        return input_ids, attention_mask, total_index


def run_embedding(json_file, outdir, batch_size=32):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    assert os.path.exists(json_file), 'Mapping file does not exist: {}'.format(json_file)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    dataset = ToyDataset(json_file, transforms=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=12)
    
    t5_model = T5EncoderModel.from_pretrained("t5-11b").to(device)
    
    with torch.no_grad():
        for input_ids, attention_mask, index in tqdm(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            with torch.no_grad():
                outputs = t5_model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_states = outputs.last_hidden_state 
            attention_mask = attention_mask.bool()
            last_hidden_states = last_hidden_states.masked_fill(~attention_mask.unsqueeze(-1), 0)
            for i, idx in enumerate(index):
                t5_emb = last_hidden_states[i].cpu().numpy()
                t5_emb_path = os.path.join(outdir, '{}.t5_emb'.format(idx))
                with open(t5_emb_path, 'wb') as t5_bf:
                    t5_bf.write(t5_emb.tobytes())

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    run_embedding(args.json_file, args.outdir, args.batch_size)