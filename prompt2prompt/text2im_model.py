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
# https://github.com/openai/glide-text2im/blob/main/glide_text2im/text2im_model.py
# ------------------------------------------------------------------------------------------------

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from model.layers import timestep_embedding, LayerNorm
from .unet import UNetModel

class Text2ImUNet(UNetModel):
    """
    A UNetModel that conditions on text with an encoding transformer.

    Expects an extra kwarg `tokens` of text.

    :param text_ctx: number of text tokens to expect.
    :param xf_width: hidden channels of text encodings.
    :param xf_final_ln: whether to apply a final layer norm to text encodings.
    :param use_pretrained_text_encoder: whether to use a pretrained text encoder.
    """

    def __init__(
        self,
        text_ctx: int,
        xf_width: int,
        xf_final_ln: bool,
        *args,
        clip_embed_dim:int=768,
        use_clip_emb:bool=False,
        use_pretrained_text_encoder:bool=False,
        **kwargs,
    ):
        self.text_ctx = text_ctx
        self.xf_width = xf_width
        self.clip_embed_dim = clip_embed_dim 
        if "noise_cond_augment" in kwargs:
            del kwargs['noise_cond_augment']

        if not xf_width:
            super().__init__(*args, **kwargs, encoder_channels=None)
        else: 
            super().__init__(*args, **kwargs, encoder_channels=xf_width)

        self.use_pretrained_text_encoder = use_pretrained_text_encoder
        self.use_clip_emb = use_clip_emb
        self.clip_embed_proj = nn.Linear(self.clip_embed_dim, self.model_channels * 4)
        self.final_ln = LayerNorm(xf_width) if xf_final_ln else nn.Identity()

        if xf_width > 0:
            self.transformer_proj = nn.Linear(xf_width, self.model_channels * 4)
        self.time_to_half = nn.Linear(self.model_channels * 4, self.model_channels * 2)
        self.clip_to_half = nn.Linear(self.model_channels * 4, self.model_channels * 2)

    def convert_to_fp16(self):
        super().convert_to_fp16()
        if not self.use_pretrained_text_encoder and self.xf_width > 0:
            self.transformer_proj.to(th.float16)

    def forward(
        self, 
        x, 
        timesteps, 
        clip_emb=None, 
        text_encodings=None,
        attn_controller=None,
        **kwargs
    ):
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels).type(self.dtype))
        
        text_encodings = self.final_ln(text_encodings)
        text_embeds = F.avg_pool1d(text_encodings.transpose(1, 2), kernel_size=text_encodings.size(1)).squeeze(-1)
        xf_proj = self.transformer_proj(text_embeds) # (B, 4 * C)
        xf_out = text_encodings.permute(0, 2, 1)

        if not self.use_clip_emb:
            # combine time_emb & text_emb
            emb = th.cat([self.time_to_half(emb), self.clip_to_half(xf_proj)], dim=1).to(emb)
        else:
            # combine time_emb & clip_emb & text_emb
            emb = th.cat([self.time_to_half(emb + xf_proj.to(emb) + clip_emb),
                        self.clip_to_half(clip_emb)], dim=1).to(clip_emb)
    
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, xf_out, attn_controller=attn_controller)
            hs.append(h)
        h = self.middle_block(h, emb, xf_out, attn_controller=attn_controller)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, xf_out, attn_controller=attn_controller)
        h = self.out(h)
        
        attn_controller.update()
        return h
    

def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult,
    attention_resolutions,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    use_pretrained_text_encoder,
    use_clip_emb,
    text_ctx,
    xf_width,
    xf_final_ln,
    resblock_updown,
    use_fp16,
    learn_sigma,
    noise_cond_augment,
    **kwargs,
):
    if channel_mult == "":
        if image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    if attention_resolutions != "":
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))

    return Text2ImUNet(
        text_ctx=text_ctx,
        xf_width=xf_width,
        xf_final_ln=xf_final_ln,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        noise_cond_augment=noise_cond_augment,
        use_pretrained_text_encoder=use_pretrained_text_encoder,
        use_clip_emb=use_clip_emb,
    )