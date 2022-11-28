import csv
import base64
from email.mime import base
import cv2
import numpy as np 
import torch
import clip
import torchvision.transforms as T
from transformers import T5Tokenizer, T5EncoderModel
from utils.tokenizer import get_encoder
from PIL import Image
import os
import json
from einops import rearrange

# for T5 embedding
t5_tokenizer = T5Tokenizer.from_pretrained("t5-11b")
t5_model = T5EncoderModel.from_pretrained("t5-11b")
t5_model = t5_model.cuda()
t5_model.eval()

# for CLIP embedding
clip_model, preprocess = clip.load("ViT-L/14")
clip_model = clip_model.cuda()
clip_model.eval()


# CLIP embedding
def get_clip_text_embedding(text):
    tokenizer = get_encoder(text_ctx=256)
    token, mask = tokenizer(text)
    token, mask = map(lambda t: torch.tensor(t).unsqueeze(0), [token, mask])
    
    token4clip = clip.tokenize(text, truncate=True).cuda()
    text_emb = clip_model.encode_text(token4clip)
    
    text_emb, token, mask = map(lambda t: t.cuda(), [text_emb, token, mask])

    return text_emb, token, mask


def get_clip_image_embedding(img_path):
    img = Image.open(img_path)
    img = preprocess(img)
    img = img.cuda()
    img_emb = clip_model.encode_image(img.unsqueeze(0))
    return img_emb



def get_t5_text_encoding(text):
    text = [text]
    encoding = t5_tokenizer(text,
                            padding='max_length',
                            max_length=256,
                            truncation=True,
                            return_tensors="pt")
    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
    with torch.no_grad():
        outputs = t5_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state  # [b, 256, 1024]
    attention_mask = attention_mask.bool()
    # just force all embeddings that is padding to be equal to 0.
    last_hidden_states = last_hidden_states.masked_fill(~rearrange(attention_mask, '... -> ... 1'), 0.)
    num_valid_tokens = attention_mask.sum(axis=1).cpu()  # [b]

    return last_hidden_states, num_valid_tokens


def numpy2base64(np_arr):
    bytes = np_arr.tobytes()
    encoded_str = base64.b64encode(bytes).decode('utf-8')    
    return encoded_str


def base642numpy(base64_str, np_dtype):
    decoded = base64.b64decode(base64_str)
    np_arr = np.frombuffer(decoded, np_dtype)
    return np_arr


def base64_from_img(img_array):
    _encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    try:
        img = cv2.imencode('.jpg', img_array, _encode_params)[1].tobytes()
        img_base64 = base64.b64encode(img).decode('utf-8')
        return img_base64
    except:
        return None


class TSVWriter(object):
    def __init__(self, tsv_path):
        if tsv_path.endswith(".tsv"):
            tsv_basename = os.path.splitext(tsv_path)[0]
        else:
            tsv_basename = tsv_path
        self.tsv_img_path         = tsv_basename + '.tsv'
        self.lineidx_path         = tsv_basename + '.lineidx'
        self._tsv_img             = None
        self._lineidx             = None
        self.img_writer           = None
        self.lineidx_writer       = None

        self._open_file()

    def _open_file(self):
        if not os.path.isfile(self.tsv_img_path) or os.path.getsize(self.tsv_img_path) == 0:
            mode = "w+"
        else:
            mode = "w+"

        self._tsv_img       = open(self.tsv_img_path, mode, encoding="utf-8")
        self._lineidx       = open(self.lineidx_path, mode, encoding="utf-8")
        self.img_writer     = csv.writer(self._tsv_img, delimiter="\t",quoting=csv.QUOTE_NONE, quotechar='')
        self.lineidx_writer = csv.writer(self._lineidx, delimiter="\t",quoting=csv.QUOTE_NONE, quotechar='')

    def append(self, data_id, json_info, t5_base64, clip_text_base64, clip_text_token_base64):
        """encode data to TSV, and record pos to lineidx_file"""
        res, log_info = False, ""
        offset = self._tsv_img.tell()
        try:
            # self.img_writer.writerow([data_id, json_info, img_base64])
            # self.lineidx_writer.writerow([offset])
            self._tsv_img.write("\t".join([str(data_id), json_info, t5_base64, clip_text_base64, clip_text_token_base64])+"\n")
            self._lineidx.write(str(offset)+"\n")
            res = True
        except Exception as e:
            log_info = f"tsv_writer wrote fail with {e} at {offset}"
        return res, log_info

    def close(self):
        self._tsv_img.close()
        self._lineidx.close()
        self._tsv_img = None
        self._lineidx = None


output_path = "sr_eval.tsv"
tsv_writer = TSVWriter(output_path)

with open("sr_eval_prompts.tsv") as file:
    prompt_file = csv.reader(file, delimiter="\t")
    for idx, line in enumerate(prompt_file):
        if idx == 0:
            continue
        text, img_path = line

        raw_image = cv2.imread(img_path)
        # img_array =  cv2.cvtColor(raw_image,cv2.COLOR_BGR2RGB)
        img_base64 = base64_from_img(raw_image)

        # t5
        t5_encodings, t5_num_valid_tokens = get_t5_text_encoding(text)
        t5_encodings = t5_encodings.detach().cpu().numpy()[0]  # fp32, [256, 1024]
        t5_num_valid_tokens = t5_num_valid_tokens.numpy()[0]   # int

        # clip
        clip_text_emb, clip_text_token, clip_mask = get_clip_text_embedding(text)
        clip_text_emb = clip_text_emb.detach().cpu().numpy()[0]          # fp16, [768]
        clip_text_token = clip_text_token.detach().cpu().numpy()[0]      # int,  [256]
        clip_num_valid_tokens = clip_mask.sum(axis=-1).cpu().numpy()[0]  # int


        meta_info = {'text': text,
                     't5_num_valid_tokens': str(t5_num_valid_tokens),
                     'clip_num_valid_tokens': str(clip_num_valid_tokens),
                     'img': img_base64,
                     }
        json_str = json.dumps(meta_info) 
        encoded = list(map(lambda x: numpy2base64(x), [t5_encodings, clip_text_emb, clip_text_token]))
        decoded = [base642numpy(encoded[0], np.float32).reshape(256, 1024),
                   base642numpy(encoded[1], np.float16),
                   base642numpy(encoded[2], np.int64)]
        # import pdb; pdb.set_trace()
        tsv_writer.append(idx, json_str, *encoded)
