# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Code was based on https://github.com/xinyu1205/recognize-anything/tree/main
# reference: https://arxiv.org/abs/2306.03514

import yaml
import numpy as np

from paddle import nn
import paddle
from paddlenlp.transformers import BertTokenizer
from paddle.nn import functional as F
from paddle.nn.initializer import Constant
from paddlenlp.transformers.bert.configuration import BertConfig

from ..backbone.model_zoo.vision_transformer import VisionTransformer
from .bert import BertModel, BertLMHeadModel
from ..clip.clip import CLIP_DICT, tokenize
from ..clip.tokenizer import Tokenizer
from ..backbone.legendary_models.swin_transformer import SwinTransformer


class RamVis(VisionTransformer):
    def forward_features(self, x):
        return x


class RamSwin(SwinTransformer):
    def forward_features(self, x):
        x, output_dimensions = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x, output_dimensions = layer(x, output_dimensions)

        x = self.norm(x)  # B L C
        x_cls = self.avgpool(x.transpose([0, 2, 1]))  # B C 1
        return paddle.concat([x_cls.transpose([0, 2, 1]), x], axis=1)

    def forward(self, x):
        x = self.forward_features(x)
        return x


def RamSwin_large_patch4_window12_384():
    return RamSwin(
        img_size=384,
        patch_size=4,
        in_chans=3,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False)


def RamSwin_large_patch4_window7_224():
    return RamSwin(
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False)


def RamSwin_base_patch4_window12_384():
    return RamSwin(
        img_size=384,
        patch_size=4,
        in_chans=3,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False)


def RamSwin_base_patch4_window7_224():
    return RamSwin(
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False)


CONFIG_PATH = 'ppcls'


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


def build_text_embed(model_clip, tokenizer, caption):
    with paddle.no_grad():
        texts = tokenize(caption, tokenizer)
        text_embeddings = model_clip.encode_text(texts)
        text_embeddings /= text_embeddings.norm(axis=-1, keepdim=True)
    return text_embeddings


class AsymmetricLoss(nn.Layer):
    def __init__(self,
                 gamma_neg=4,
                 gamma_pos=1,
                 clip=0.05,
                 eps=1e-8,
                 disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = nn.functional.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clip(max=1)

        # Basic CE calculation
        los_pos = y * paddle.log(xs_pos.clip(min=self.eps))
        los_neg = (1 - y) * paddle.log(xs_neg.clip(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                paddle.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = paddle.pow(1 - pt, one_sided_gamma.astype(pt.dtype))
            if self.disable_torch_grad_focal_loss:
                paddle.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


def init_tokenizer(tokenizer_name="bert-base-uncased"):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def read_yaml(rpath):
    with open(rpath, 'r') as f:
        return yaml.safe_load(f)


def create_vit(vit,
               image_size,
               use_grad_checkpointing=False,
               ckpt_layer=0,
               drop_path_rate=0):

    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit == 'base':
        vision_width = 768
        visual_encoder = RamVis(
            img_size=image_size,
            patch_size=16,
            embed_dim=vision_width,
            depth=12,
            class_num=0,
            num_heads=12,
            use_grad_checkpointing=use_grad_checkpointing,
            ckpt_layer=ckpt_layer,
            drop_path_rate=0 or drop_path_rate)
    elif vit == 'large':
        vision_width = 1024
        visual_encoder = RamVis(
            img_size=image_size,
            patch_size=16,
            class_num=0,
            embed_dim=vision_width,
            depth=24,
            num_heads=16,
            use_grad_checkpointing=use_grad_checkpointing,
            ckpt_layer=ckpt_layer,
            drop_path_rate=0.1 or drop_path_rate)
    return visual_encoder, vision_width


class RAM(nn.Layer):
    def __init__(self,
                 med_config='',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 prompt='',
                 clip_pretraind=None,
                 threshold=0.68,
                 delete_tag_index=[],
                 tag_list='',
                 tag_list_chinese='',
                 clip_version='',
                 q2l_config='',
                 ram_class_threshold_path='',
                 pretrained='',
                 stage='eval'):

        super().__init__()

        # create image encoder
        self.stage = stage
        if self.stage == 'train':
            assert clip_pretraind
            self.clip_tokenizer = Tokenizer()
            assert clip_version in CLIP_DICT.keys(
            ), 'please check the clip structure'
            self.CLIP = CLIP_DICT[clip_version]
            params = paddle.load(clip_pretraind)
            self.CLIP.set_state_dict(params)
            self.CLIP.eval()

        if vit == 'swin_b':
            vision_width = 1024
            if image_size == 224:
                self.visual_encoder = RamSwin_base_patch4_window7_224()
            elif image_size == 384:
                self.visual_encoder = RamSwin_base_patch4_window12_384()

        elif vit == 'swin_l':
            vision_width = 1536
            if image_size == 224:
                self.visual_encoder = RamSwin_large_patch4_window7_224()
            elif image_size == 384:
                self.visual_encoder = RamSwin_large_patch4_window12_384()

        else:
            self.visual_encoder, vision_width = create_vit(
                vit, image_size, vit_grad_ckpt, vit_ckpt_layer)

        # create tokenzier
        self.tokenizer = init_tokenizer()

        # Tag2Text employ encoder-decoder architecture for image-tag-text generation: image-tag interaction encoder and image-tag-text decoder
        # create image-tag interaction encoder
        encoder_config = BertConfig.from_dict(read_yaml(med_config))
        encoder_config.encoder_width = 512
        self.tag_encoder = BertModel(
            config=encoder_config, add_pooling_layer=False)

        # create image-tag-text decoder
        decoder_config = BertConfig.from_dict(read_yaml(med_config))
        self.text_decoder = BertLMHeadModel(config=decoder_config)

        self.delete_tag_index = delete_tag_index
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

        # load tag list
        self.tag_list = self.load_tag_list(tag_list)
        self.tag_list_chinese = self.load_tag_list(tag_list_chinese)

        # create image-tag recognition decoder
        self.threshold = threshold
        self.num_class = len(self.tag_list)
        q2l_config = BertConfig.from_dict(read_yaml(q2l_config))
        q2l_config.encoder_width = 512
        self.tagging_head = BertModel(
            config=q2l_config, add_pooling_layer=False)
        self.tagging_head.resize_token_embeddings(len(self.tokenizer))
        self.label_embed = self.create_parameter(
            shape=(self.num_class, q2l_config.encoder_width),
            default_initializer=Constant())

        if q2l_config.hidden_size != 512:
            self.wordvec_proj = nn.Linear(512, q2l_config.hidden_size)
        else:
            self.wordvec_proj = nn.Identity()

        self.fc = nn.Linear(q2l_config.hidden_size, 1)

        self.del_selfattention()

        self.tagging_loss_function = AsymmetricLoss(
            gamma_neg=7, gamma_pos=0, clip=0.05)

        self.image_proj = nn.Linear(vision_width, 512)

        # adjust thresholds for some tags
        self.class_threshold = paddle.ones([self.num_class]) * self.threshold
        ram_class_threshold_path = ram_class_threshold_path
        with open(ram_class_threshold_path, 'r', encoding='utf-8') as f:
            ram_class_threshold = [float(s.strip()) for s in f]
        for key, value in enumerate(ram_class_threshold):
            self.class_threshold[key] = value

    def load_tag_list(self, tag_list_file):
        with open(tag_list_file, 'r', encoding='utf-8') as f:
            tag_list = f.read().splitlines()
        tag_list = np.array(tag_list)
        return tag_list

    # delete self-attention layer of image-tag recognition decoder to reduce computation, follower Query2Label
    def del_selfattention(self):
        del self.tagging_head.embeddings
        for layer in self.tagging_head.encoder.layer:
            del layer.attention

    def forward(self,
                image_ram,
                text=None,
                image_tag=None,
                tag_input_tokenzier=None,
                image_clip=None):
        """
        image-》 image_ram
        image224 -> image_clip
        call function as forward

        Args:
            image: type: paddle.Tensor  shape: batch_size * 3 * 384 * 384
            caption: type: paddle.Tensor  len: batch_size * embedding_size
            tag: type: paddle.Tensor   shape: batch * class_num (e.g. 3429)   value: positive sample is 1.0, negative sample is 0.0

        Returns:
            loss: type: paddle.Tensor
        """
        assert self.stage == 'train'
        label_embed = nn.functional.relu(self.wordvec_proj(self.label_embed))
        clip_feature = self.CLIP.encode_image(image_clip)

        image_embeds = self.image_proj(self.visual_encoder(image_ram))
        image_atts = paddle.ones(
            paddle.shape(image_embeds)[:-1], dtype=paddle.int32)

        ##================= Distillation from CLIP ================##
        image_cls_embeds = image_embeds[:, 0, :]
        image_spatial_embeds = image_embeds[:, 1:, :]

        loss_dis = 0.
        if isinstance(clip_feature, paddle.Tensor):
            loss_dis = F.l1_loss(image_cls_embeds, clip_feature)

##================= Image Tagging ================##
        bs = paddle.shape(image_embeds)[0]
        #label_embed = paddle.repeat_interleave(label_embed.unsqueeze(0),[bs, 1, 1])
        label_embed = label_embed.unsqueeze(0).tile([bs, 1, 1]).squeeze(1)

        tagging_embed = self.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging', )

        logits = self.fc(tagging_embed[0]).squeeze(-1)
        loss_tag = 0.
        if isinstance(image_tag, paddle.Tensor):
            loss_tag = self.tagging_loss_function(logits, image_tag)

        ##================= Image-Tag-Text Generation ================##
        encoder_input_ids = tag_input_tokenzier.get("input_ids")
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

        # put input tag into image-tag interaction encoder to interact with image embeddings
        output_tagembedding = self.tag_encoder(
            encoder_input_ids,
            attention_mask=tag_input_tokenzier.get("attention_mask"),
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True, )

        decoder_input_ids = text.get("input_ids")
        decoder_input_ids[:, 0] = self.tokenizer.enc_token_id

        decoder_targets = masked_fill(
            decoder_input_ids,
            decoder_input_ids == self.tokenizer.pad_token_id, -100)
        decoder_targets[:, :self.prompt_length] = -100

        decoder_output = self.text_decoder(
            decoder_input_ids,
            attention_mask=text.get("attention_mask"),
            encoder_hidden_states=output_tagembedding.last_hidden_state,
            encoder_attention_mask=None,
            labels=decoder_targets,
            return_dict=True, )

        loss_t2t = decoder_output.loss

        return loss_t2t, loss_tag, loss_dis

    # to support paddle framework
    def inference(
            self,
            image,
            threshold=0.4,
            tag_input=None, ):

        label_embed = F.relu(self.wordvec_proj(self.label_embed))

        image_embeds = self.image_proj(self.visual_encoder(image))
        image_atts = paddle.ones(image_embeds.shape[:-1], dtype=paddle.int32)

        # recognized image tags using image-tag recogntiion decoder
        image_cls_embeds = image_embeds[:, 0, :]
        image_spatial_embeds = image_embeds[:, 1:, :]

        bs = image_spatial_embeds.shape[0]
        label_embed = label_embed.unsqueeze(0).tile([bs, 1, 1]).squeeze(1)
        tagging_embed = self.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging', )

        logits = self.fc(tagging_embed[0]).squeeze(-1)

        return logits

    def generate_tag_openset(
            self,
            image,
            threshold=0.68,
            tag_input=None, ):

        label_embed = F.relu(self.wordvec_proj(self.label_embed))

        image_embeds = self.image_proj(self.visual_encoder(image))
        image_atts = paddle.ones(
            [image_embeds.size()[:-1]], dtype=paddle.int32)

        # recognized image tags using image-tag recogntiion decoder
        image_cls_embeds = image_embeds[:, 0, :]
        image_spatial_embeds = image_embeds[:, 1:, :]

        bs = image_spatial_embeds.shape[0]
        label_embed = label_embed.unsqueeze(0).repeat([bs, 1, 1])
        tagging_embed = self.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging', )

        logits = self.fc(tagging_embed[0]).squeeze(-1)

        return logits


# load RAM pretrained model parameters
def ram(pretrained='', **kwargs):
    model = RAM(pretrained='', **kwargs)
    return model
