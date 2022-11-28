# -*- coding: utf-8 -*-
# @Author: Yihao Chen
# @Date:   2022-05-24 11:07:42
# @Last Modified by:   Yihao Chen
# @Last Modified time: 2022-06-21 17:13:22

import os
import numpy as np

class BinFile(object):
    def __init__(self, bin_file, map_file):
        self.bin_file = bin_file
        self.map = map_file
        self._fp = None
        self._map = None

        self._ensure_map_loaded()

    def num_rows(self):
        return len(self._map)

    def seek(self, idx):
        self._ensure_bin_opened()
        pos = self._map[idx][0]
        if len(self._map[idx]) == 3:
            length = self._map[idx][1]
            tsv_idx = self._map[idx][2]
            dtype = np.float32
        else:
            length = 3072
            tsv_idx = self._map[idx][1]
            dtype = np.float16

        self._fp.seek(pos)
        array = np.frombuffer(self._fp.read(length), dtype=dtype)
        return array, tsv_idx
        # split = array.size // 2
        # return array[:split], array[split:], self._map[idx][1]

    def close(self):
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def _ensure_map_loaded(self):
        if self._map is None:
            with open(self.map, 'r') as fp:
                lines = fp.readlines()
                self._map = [[int(i) for i in line.strip().split("\t")] for line in lines]

                # self._map = []
                # fpos = 0
                # fsize = os.fstat(fp.fileno()).st_size
                # while fpos != fsize:
                #     con = fp.readline()
                #     splits = [int(i) for i in con.strip().split("\t")]
                #     self._map.append(splits)
                #     fpos = fp.tell()

    def _ensure_bin_opened(self):
        if self._fp is None:
            self._fp = open(self.bin_file, 'rb')


if __name__ == '__main__':
    bin_file = "/comp_robot/cv_public_dataset/clip_embedding/t5_cc3m_train_resize.bin"
    map_file = "/comp_robot/cv_public_dataset/clip_embedding/t5_cc3m_train_resize.txt"
    bin = BinFile(bin_file, map_file)
    
    import pdb; pdb.set_trace()

    embedding, index = bin.seek(0)

    embedding = embedding.reshape(-1, 1024)



    # data = np.array([[1., 2.], [3., 4.], [5., 6.]], dtype=np.float32)
    # print(data)
    # print(data.shape)
    # data2 = np.frombuffer(data.tobytes(), dtype=np.float32)
    # data2 = data2.reshape(-1, 2)
    # print(data2)
    # print(data2.shape)


