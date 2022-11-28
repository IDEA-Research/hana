
import torch
import clip
import sys
sys.path.insert(0,'..')
from clip_embedding import get_dataset
from transformers import T5Tokenizer, T5EncoderModel


def test_clip_and_t5_dataset():
    ds = get_dataset('CC_3M', text_encoder='T5')
    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    clip_model, _ = clip.load("ViT-L/14", device=device)
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-11b")
    t5_model = T5EncoderModel.from_pretrained("t5-11b")
    if device == "cuda":
        t5_model = t5_model.cuda()

    max_diff = lambda x, y: ((x.cpu() - y.cpu()).abs()).max()
    
    for clip_image_embedding, clip_text_embedding, img, text, t5_text_encodings in dl:
        encoding = t5_tokenizer(text,
                            padding='max_length',
                            max_length=256,
                            truncation=True,
                            return_tensors="pt")

        input_ids, mask = encoding.input_ids, encoding.attention_mask
        if device == "cuda":
            input_ids = input_ids.cuda()
        valid_text_encodings = int(mask.sum())
        outputs = t5_model(input_ids=input_ids)
        otf_t5_text_encodings = outputs.last_hidden_state  # [b, 256, 1024]
        text = clip.tokenize(list(text)).to(device)
        otf_clip_text_embedding = clip_model.encode_text(text)
        print('max t5 diff', max_diff(otf_t5_text_encodings[:, :valid_text_encodings+1],
                                      t5_text_encodings[:, :valid_text_encodings+1]))
        print('max clip diff', max_diff(otf_clip_text_embedding, clip_text_embedding))
        import pdb; pdb.set_trace()
        break



if __name__ == '__main__':
    test_clip_and_t5_dataset()

    