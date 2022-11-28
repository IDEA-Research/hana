# -*- coding: utf-8 -*-
# @Author: Yihao Chen
# @Date:   2022-05-24 11:07:42
# @Last Modified by:   Yihao Chen
# @Last Modified time: 2022-09-14 20:13:47

import json
import os
import numpy as np

class CombineBinFile(object):
    def __init__(self, t5_bin_file, t5_map_file, clip_bin_file, clip_map_file):
        self.t5_bin_file = t5_bin_file
        self.t5_map_file = t5_map_file
        self.clip_bin_file = clip_bin_file
        self.clip_map_file = clip_map_file
        
        self._t5_fp = None
        self._t5_map = None
        self._clip_fp = None
        self._clip_map = None
        
        self.overlad_cache_file = os.path.splitext(t5_bin_file)[0] + ".cache"

        self._ensure_map_loaded()

    def num_rows(self):
        return len(self._t5_map)

    def seek(self, idx):
        self._ensure_bin_opened()

        assert self._t5_map[idx][2] == self._clip_map[idx][1]
        tsv_idx = self._clip_map[idx][1]

        pos = self._t5_map[idx][0]
        length = self._t5_map[idx][1]
        dtype = np.float32

        self._t5_fp.seek(pos)
        t5_array = np.frombuffer(self._t5_fp.read(length), dtype=dtype)

        pos = self._clip_map[idx][0]        
        length = 3072
        dtype = np.float16

        self._clip_fp.seek(pos)
        clip_array = np.frombuffer(self._clip_fp.read(length), dtype=dtype)

        return t5_array, clip_array, tsv_idx

    def close(self):
        if self._t5_fp is not None:
            self._t5_fp = None
            self._t5_map = None
            self._clip_fp = None
            self._clip_map = None

    def _ensure_map_loaded(self):
        if self._t5_map is None and not os.path.exists(self.overlad_cache_file):
            with open(self.t5_map_file, 'r') as fp:
                lines = fp.readlines()
                t5_map = [[int(i) for i in line.strip().split("\t")] for line in lines]

            with open(self.clip_map_file, 'r') as fp:
                lines = fp.readlines()
                clip_map = [[int(i) for i in line.strip().split("\t")] for line in lines]

            t5_dict = {m[-1] : m for m in t5_map if len(m) == 3}
            clip_dict = {m[-1] : m for m in clip_map if len(m) == 2}

            self._t5_map = []
            self._clip_map = []
            for key in t5_dict:
                if key in clip_dict:
                    self._t5_map.append(t5_dict[key])
                    self._clip_map.append(clip_dict[key])
            
            with open(self.overlad_cache_file, 'w') as fp:
                data = dict(
                    t5=self._t5_map,
                    clip=self._clip_map
                )
                fp.write(json.dumps(data))
        else:
            with open(self.overlad_cache_file, 'r') as fp:
                data = json.loads(fp.readline())
                if data is not None:
                    self._t5_map = data['t5']
                    self._clip_map = data['clip']
                else:
                    raise RuntimeError("Cache file is empty")

    def _ensure_bin_opened(self):
        if self._t5_fp is None:
            self._t5_fp = open(self.t5_bin_file, 'rb')

        if self._clip_fp is None:
            self._clip_fp = open(self.clip_bin_file, 'rb')


if __name__ == '__main__':
    t5_bin_file = "/comp_robot/cv_public_dataset/clip_embedding/t5_cc3m_train_resize.bin"
    t5_map_file = "/comp_robot/cv_public_dataset/clip_embedding/t5_cc3m_train_resize.txt"

    clip_bin_file = "/comp_robot/cv_public_dataset/ConceptualCaptions/CLIP_ViT_L_embedding.bin"
    clip_map_file = "/comp_robot/cv_public_dataset/ConceptualCaptions/CLIP_ViT_L_embedding.txt"

    bin = CombineBinFile(t5_bin_file, t5_map_file, clip_bin_file, clip_map_file)

    t5_embedding, clip_embedding, index = bin.seek(0)
    embedding = t5_embedding.reshape(-1, 1024)
