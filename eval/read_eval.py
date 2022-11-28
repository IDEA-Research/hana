import os.path as op
import json
import base64
import numpy as np
from torch.utils.data import Dataset
import os
from idea.dataset.tsv import generate_lineidx, FileProgressingbar
import torchvision.transforms as T
from io import BytesIO
from PIL import Image
from clip_embedding import SR_ImageProcessing

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def base642numpy(base64_str, np_dtype):
    decoded = base64.b64decode(base64_str)
    np_arr = np.frombuffer(decoded, np_dtype)
    return np_arr


def img_from_base64(imagestring):
    jpgbytestring = base64.b64decode(imagestring)
    image = BytesIO(jpgbytestring)
    image = Image.open(image).convert("RGB")
    return image


class TSVFile(object):
    def __init__(self, tsv_file, lineidx_file=None, silence=True):
        self.tsv_file = tsv_file
        if lineidx_file is None:
            self.lineidx = op.splitext(tsv_file)[0] + '.lineidx'
        else:
            self.lineidx = lineidx_file
        self._fp = None
        self._lineidx = None
        self.silence = silence
        self.count = 0
        self._ensure_lineidx_loaded()

    def num_rows(self):
        return len(self._lineidx)

    def seek(self, idx):
        self._ensure_tsv_opened()
        pos = self._lineidx[idx]
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]

    def seek_list(self, idxs, q):
        assert isinstance(idxs, list)
        self._ensure_tsv_opened()
        for idx in idxs:
            pos = self._lineidx[idx]
            self._fp.seek(pos)
            q.put([s.strip() for s in self._fp.readline().split('\t')])

    def close(self):
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def _ensure_lineidx_loaded(self):
        if not op.isfile(self.lineidx) and not op.islink(self.lineidx):
            generate_lineidx(self.tsv_file, self.lineidx)

        if self._lineidx is None:
            with open(self.lineidx, 'r') as fp:
                if not self.silence:
                    bar = FileProgressingbar(fp, "Loading lineidx {0}: ".format(self.lineidx))
                self._lineidx = []
                fpos = 0
                fsize = os.fstat(fp.fileno()).st_size
                while fpos != fsize:
                    i = fp.readline()
                    fpos = fp.tell()
                    self._lineidx.append(int(i.strip()))
                    if not self.silence:
                        bar.update()

    def _ensure_tsv_opened(self):
        if self._fp is None:
            self._fp = open(self.tsv_file, 'r')



class EvalDataset(Dataset):
    '''
        TSV dataset for tsv file
    '''    
    def __init__(self, tsv_file, lineidx_file=None, repeat_time=1):
        self.tsv = TSVFile(tsv_file, lineidx_file)
        self.repeat_time = repeat_time

    @property
    def real_len(self):
        return self.tsv.num_rows()

    def check(self, row):
        return True

    def read_embeddings(self, row):
        try:
            t5_embedding, clip_text_emb, clip_text_token = row[2], row[3], row[4],
            decoded = [base642numpy(t5_embedding, np.float32).reshape(256, 1024),
                       base642numpy(clip_text_emb, np.float16),
                       base642numpy(clip_text_token, np.int64)]
            return decoded
        except Exception as e:
            return False

    def read_meta(self, row):
        info = json.loads(row[1])
        return info

    def __getitem__(self, index):
        index = index % self.real_len
        row = self.tsv.seek(index)
        embeddings = self.read_embeddings(row)
        meta_info = self.read_meta(row)
        return embeddings, meta_info

    def __len__(self):
        return int(self.tsv.num_rows() * self.repeat_time)

    def get_row(self, index):
        return self.__getitem__(index)


class SREvalDataset(Dataset):
    '''
        TSV dataset for tsv file
    '''    
    def __init__(self, tsv_file, lineidx_file, config, HR_transforms, LR_transforms, repeat_time=1):
        self.tsv = TSVFile(tsv_file, lineidx_file)
        self.repeat_time = repeat_time
        self.sr_processor = SR_ImageProcessing(config, HR_transforms, LR_transforms)

    @property
    def real_len(self):
        return self.tsv.num_rows()

    def check(self, row):
        return True

    def read_embeddings(self, row):
        try:
            t5_embedding, clip_text_emb, clip_text_token = row[2], row[3], row[4],
            decoded = [base642numpy(t5_embedding, np.float32).reshape(256, 1024),
                       base642numpy(clip_text_emb, np.float16),
                       base642numpy(clip_text_token, np.int64)]
            return decoded
        except Exception as e:
            return False

    def read_meta(self, row):
        info = json.loads(row[1])
        return info

    def __getitem__(self, index):
        index = index % self.real_len
        row = self.tsv.seek(index)
        embeddings = self.read_embeddings(row)
        meta_info = self.read_meta(row)
        assert 'img' in meta_info.keys()
        img = img_from_base64(meta_info['img'])
        img, LR_img = self.sr_processor.image_transform(img)
        del meta_info['img']

        return img, LR_img, embeddings, meta_info

    def __len__(self):
        return int(self.tsv.num_rows() * self.repeat_time)

    def get_row(self, index):
        return self.__getitem__(index)