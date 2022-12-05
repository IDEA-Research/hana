# -*- coding: utf-8 -*-
# @Author: Yihao Chen
# @Date:   2021-08-13 14:35:27
# @Last Modified by:   Yihao Chen
# @Last Modified time: 2022-05-23 15:32:27

import os
import base64
from io import BytesIO
from PIL import Image

import progressbar

class FileProgressingbar:
    fileobj = None
    pbar = None
    def __init__(self, fileobj, msg):
        fileobj.seek(0, os.SEEK_END)
        flen = fileobj.tell()
        fileobj.seek(0, os.SEEK_SET)
        self.fileobj = fileobj
        widgets = [msg, progressbar.AnimatedMarker(), ' ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
        self.pbar = progressbar.ProgressBar(widgets=widgets, maxval=flen).start()

    def update(self):
        self.pbar.update(self.fileobj.tell())


def img_from_base64(imagestring):
    jpgbytestring = base64.b64decode(imagestring)
    image = BytesIO(jpgbytestring)
    image = Image.open(image).convert("RGB")
    return image


def generate_lineidx(filein, idxout):
    assert not os.path.isfile(idxout)
    with open(filein, 'r') as tsvin, open(idxout, 'w') as tsvout:
        bar = FileProgressingbar(tsvin, 'Generating lineidx {0}: '.format(idxout))
        fsize = os.fstat(tsvin.fileno()).st_size
        fpos = 0
        while fpos != fsize:
            tsvout.write(str(fpos)+"\n")
            tsvin.readline()
            fpos = tsvin.tell()
            bar.update()
