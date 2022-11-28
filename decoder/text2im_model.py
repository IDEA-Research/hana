import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .nn import linear, timestep_embedding
from .unet import UNetModel
from .xf import LayerNorm, Transformer, convert_module_to_f16

from deepspeed import checkpointing
dsp_checkpoint = checkpointing.checkpoint


class Text2ImUNet(UNetModel):
    clip_embed_dim: int = 768
    """
    A UNetModel that conditions on text with an encoding transformer.

    Expects an extra kwarg `tokens` of text.

    :param text_ctx: number of text tokens to expect.
    :param xf_width: width of the transformer.
    :param xf_layers: depth of the transformer.
    :param xf_heads: heads in the transformer.
    :param xf_final_ln: use a LayerNorm after the output layer.
    :param tokenizer: the text tokenizer for sampling/vocab size.
    """

    def __init__(
        self,
        text_ctx,
        xf_width,
        xf_layers,
        xf_heads,
        xf_final_ln,
        tokenizer,
        *args,
        cache_text_emb=False,
        xf_ar=0.0,
        xf_padding=False,
        share_unemb=False,
        use_pretrained_text_encoder=False,
        **kwargs,
    ):
        self.text_ctx = text_ctx
        self.xf_width = xf_width
        self.xf_ar = xf_ar
        self.xf_padding = xf_padding
        self.tokenizer = tokenizer
        if "noise_cond_augment" in kwargs:
            del kwargs['noise_cond_augment']

        if not xf_width:
            super().__init__(*args, **kwargs, encoder_channels=None)
        else:  # w/o training text-encoder
            super().__init__(*args, **kwargs, encoder_channels=xf_width)

        self.use_pretrained_text_encoder = use_pretrained_text_encoder

        self.clip_embed_proj = nn.Linear(
            self.clip_embed_dim, self.model_channels * 4)

        if xf_final_ln:
            self.final_ln = LayerNorm(xf_width)
        else:
            self.final_ln = None

        # if need train text-encoder
        if not self.use_pretrained_text_encoder and self.xf_width > 0:
            self.transformer = Transformer(
                text_ctx,
                xf_width,
                xf_layers,
                xf_heads,
            )
            self.token_embedding = nn.Embedding(
                self.tokenizer.n_vocab, xf_width)
            self.positional_embedding = nn.Parameter(
                th.empty(text_ctx, xf_width, dtype=th.float32))
            self.transformer_proj = nn.Linear(
                xf_width, self.model_channels * 4)

            if self.xf_padding:
                self.padding_embedding = nn.Parameter(
                    th.empty(text_ctx, xf_width, dtype=th.float32)
                )
            if self.xf_ar:
                self.unemb = nn.Linear(xf_width, self.tokenizer.n_vocab)
                if share_unemb:
                    self.unemb.weight = self.token_embedding.weight
        # if use pre-trained text encoder
        elif self.use_pretrained_text_encoder:
            assert self.xf_width > 0, "xf_width must be > 0 if using pretrained text encoder"
            self.transformer_proj = nn.Linear(
                xf_width, self.model_channels * 4)

        self.cache_text_emb = cache_text_emb
        self.cache = None
        #
        self.time_to_half = nn.Linear(
            self.model_channels * 4, self.model_channels * 2)
        self.clip_to_half = nn.Linear(
            self.model_channels * 4, self.model_channels * 2)
        #

    def convert_to_fp16(self):
        super().convert_to_fp16()
        if not self.use_pretrained_text_encoder and self.xf_width > 0:
            self.transformer.apply(convert_module_to_f16)
            self.transformer_proj.to(th.float16)
            self.token_embedding.to(th.float16)
            self.positional_embedding.to(th.float16)
            if self.xf_padding:
                self.padding_embedding.to(th.float16)
            if self.xf_ar:
                self.unemb.to(th.float16)

    def get_text_emb(self, tokens, mask):
        assert tokens is not None

        if self.cache_text_emb and self.cache is not None:
            assert (
                tokens == self.cache["tokens"]
            ).all(), f"Tokens {tokens.cpu().numpy().tolist()} do not match cache {self.cache['tokens'].cpu().numpy().tolist()}"
            return self.cache

        xf_in = self.token_embedding(tokens.long())
        xf_in = xf_in + self.positional_embedding[None]
        if self.xf_padding:
            assert mask is not None
            xf_in = th.where(mask[..., None], xf_in, self.padding_embedding[None])
        xf_out = self.transformer(xf_in.to(self.dtype))
        if self.final_ln is not None:
            xf_out = self.final_ln(xf_out)
        xf_proj = self.transformer_proj(xf_out[:, -1])
        xf_out = xf_out.permute(0, 2, 1)  # NLC -> NCL

        outputs = dict(xf_proj=xf_proj, xf_out=xf_out)

        if self.cache_text_emb:
            self.cache = dict(
                tokens=tokens,
                xf_proj=xf_proj.detach(),
                xf_out=xf_out.detach() if xf_out is not None else None,
            )

        return outputs

    def del_cache(self):
        self.cache = None

    def forward(self, x, timesteps, clip_emb=None, tokens=None, mask=None, text_encodings=None, use_clip_emb=True,
                mode='train'):
        hs = []
        emb = self.time_embed(timestep_embedding(
            timesteps, self.model_channels).type(self.dtype))
        if self.xf_width > 0:
            if not self.use_pretrained_text_encoder:
                text_outputs = self.get_text_emb(tokens, mask)
                xf_proj, xf_out = text_outputs["xf_proj"], text_outputs["xf_out"]
            else:
                if self.final_ln is not None:
                    text_encodings = self.final_ln(text_encodings)
                text_embeds = F.avg_pool1d(text_encodings.transpose(
                    1, 2), kernel_size=text_encodings.size(1)).squeeze(-1)
                xf_proj = self.transformer_proj(text_embeds)
                xf_out = text_encodings.permute(0, 2, 1)

            if not use_clip_emb:
                # mask 10% encodings
                xf_out_mask = th.zeros(xf_out.shape[0], device=xf_out.device).type(
                    self.dtype).uniform_(0, 1) < 0.9 if mode=='train' else th.ones(xf_out.shape[0], device=xf_out.device).type(self.dtype)
                xf_out = xf_out_mask[..., None, None] * xf_out
                xf_proj = xf_out_mask[..., None] * xf_proj
                # combine time_emb & text_emb
                emb = th.cat([self.time_to_half(emb),
                              self.clip_to_half(xf_proj)], dim=1).to(emb)
                # emb = emb + xf_proj.to(emb)
            else: # Dalle config
                # mask 50% encodings
                xf_out_mask = th.zeros(xf_out.shape[0], device=xf_out.device).type(
                    self.dtype).uniform_(0, 1) < 0.5 if mode=='train' else th.ones(xf_out.shape[0], device=xf_out.device).type(self.dtype)
                xf_out = xf_out_mask[..., None, None] * xf_out
                # mask 10% clip_embed
                clip_emb = clip_emb.to(emb)
                clip_emb = self.clip_embed_proj(clip_emb)
                clip_emb_mask = th.zeros(clip_emb.shape[0], device=clip_emb.device).type(
                    self.dtype).uniform_(0, 1) < 0.9 if mode=='train' else th.ones(clip_emb.shape[0], device=clip_emb.device).type(self.dtype)
                clip_emb = clip_emb_mask[..., None] * clip_emb
                # combine time_emb & clip_emb & text_emb
                emb = th.cat([self.time_to_half(emb + xf_proj.to(emb) + clip_emb),
                            self.clip_to_half(clip_emb)], dim=1).to(clip_emb)
        else:
            clip_emb = clip_emb.to(emb)
            clip_emb = self.clip_embed_proj(clip_emb)
            clip_emb_mask = th.zeros(clip_emb.shape[0],  device=clip_emb.device).type(
                self.dtype).uniform_(0, 1) < 0.9 if mode=='train' else th.ones(clip_emb.shape[0], device=clip_emb.device).type(self.dtype)
            clip_emb = clip_emb_mask[..., None] * clip_emb
            emb = th.cat([self.time_to_half(emb + clip_emb),
                         self.clip_to_half(clip_emb)], dim=1).to(clip_emb)
            xf_out = None
            
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = dsp_checkpoint(module, h, emb, xf_out)
            # h = module(h, emb, xf_out)
            hs.append(h)
        # h = dsp_checkpoint(self.middle_block, h, emb, xf_out)
        h = self.middle_block(h, emb, xf_out)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = dsp_checkpoint(module, h, emb, xf_out)
            # h = module(h, emb, xf_out)
        h = self.out(h)
        return h


class SuperResText2ImUNet(Text2ImUNet):
    """
    A text2im model that performs super-resolution.
    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 2
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 2
        noise_cond_augment = kwargs.get("noise_cond_augment", False)
        kwargs['no_self_attention'] = True
        super().__init__(*args, **kwargs)
        del self.clip_to_half
        del self.time_to_half
        self.time_token_proj = nn.Identity()
        if noise_cond_augment:
            self.lowres_cond_time_embed = copy.deepcopy(self.time_embed)
            self.time_token_proj = nn.Linear(
                self.model_channels * 8, self.model_channels * 4)

    def forward(self, x, timesteps, aug_t=None, low_res=None):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(
            low_res, (new_height, new_width), mode="bilinear", align_corners=False
        )
        x = th.cat([x, upsampled], dim=1)

        hs = []
        t_emb = self.time_embed(timestep_embedding(
            timesteps, self.model_channels).type(self.dtype))

        if aug_t is not None:
            aug_t_emb = self.lowres_cond_time_embed(
                timestep_embedding(aug_t, self.model_channels).type(self.dtype))
            t_emb = th.cat([t_emb, aug_t_emb], dim=-1)
        emb = self.time_token_proj(t_emb)

        h = x.type(self.dtype)

        for module in self.input_blocks:
            h = module(h, emb)
            # h = dsp_checkpoint(module, h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        # h = dsp_checkpoint(self.middle_block, h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
            # h = dsp_checkpoint(module, h, emb)
        h = self.out(h)

        return h


class InpaintText2ImUNet(Text2ImUNet):
    """
    A text2im model which can perform inpainting.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, timesteps, inpaint_image=None, inpaint_mask=None, **kwargs):
        if inpaint_image is None:
            inpaint_image = th.zeros_like(x)
        if inpaint_mask is None:
            inpaint_mask = th.zeros_like(x[:, :1])
        inverted_mask = (inpaint_mask == 0).int()
        inpaint_image = inpaint_image * inpaint_mask
        new_x = inpaint_image + inverted_mask * x
        return super().forward(
            new_x,
            timesteps,
            **kwargs,
        )


class SuperResInpaintText2ImUnet(Text2ImUNet):
    """
    A text2im model which can perform both upsampling and inpainting.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 3 + 1
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 3 + 1
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x,
        timesteps,
        inpaint_image=None,
        inpaint_mask=None,
        low_res=None,
        **kwargs,
    ):
        if inpaint_image is None:
            inpaint_image = th.zeros_like(x)
        if inpaint_mask is None:
            inpaint_mask = th.zeros_like(x[:, :1])
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(
            low_res, (new_height, new_width), mode="bilinear", align_corners=False
        )
        return super().forward(
            th.cat([x, inpaint_image * inpaint_mask,
                   inpaint_mask, upsampled], dim=1),
            timesteps,
            **kwargs,
        )
