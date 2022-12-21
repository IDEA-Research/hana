# coding=utf-8
# Copyright (c) 2022 IDEA. All rights reserved.
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
# Modified from:
# https://github.com/openai/glide-text2im/blob/main/glide_text2im/unet.py
# ------------------------------------------------------------------------------------------------

from copy import deepcopy
from collections import defaultdict

class AttentionControl(object):
    """
    A base class for attention control modules.
    """

    def __init__(self):
        self.cur_step = 0
        self.storage = defaultdict(list)
        self.store_mode = False
    
    def switch_to_write(self):
        self.reset()
        self.store_mode = True
        
    def switch_to_read(self):
        self.cur_step = 0
        self.store_mode = False
        # copy the storage to avoid the change of the storage during the read process
        self.storage_copy = deepcopy(self.storage)
    
    def append(self, attn):
        assert self.store_mode is True, "AttentionControl is not in write mode."
        self.storage[self.cur_step].append(attn)
    
    def pop(self):
        assert self.store_mode is False, "AttentionControl is not in read mode."
        return self.storage_copy[self.cur_step].pop(0)
    
    def update(self):
        self.cur_step += 1
    
    def reset(self):
        self.cur_step = 0
        self.storage = defaultdict(list)
        self.store_mode = False