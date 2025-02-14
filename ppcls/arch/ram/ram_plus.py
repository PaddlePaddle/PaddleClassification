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
# reference: https://arxiv.org/abs/2310.15200

import paddle
import numpy as np
from paddle import nn
from paddle.nn import functional as F
from paddle.nn.initializer import Constant

from .bert import BertModel, BertLMHeadModel
from ..clip.clip import tokenize
from .ram import RAM, AsymmetricLoss


def build_text_embed(model_clip, texts):
    with paddle.no_grad():
        text_embeddings = model_clip.encode_text(texts)
        text_embeddings /= text_embeddings.norm(axis=-1, keepdim=True)
    return text_embeddings


class RAM_plus(RAM):
    def __init__(self,
                 med_config='',
                 image_size=None,
                 vit='',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 prompt='',
                 threshold=0.68,
                 delete_tag_index=[],
                 clip_pretraind='',
                 tag_list='',
                 tag_list_chinese='',
                 clip_version='',
                 q2l_config='',
                 ram_class_threshold_path='',
                 pretrained='',
                 stage='eval'):

        super().__init__(
            med_config=med_config,
            image_size=image_size,
            vit=vit,
            vit_grad_ckpt=vit_grad_ckpt,
            vit_ckpt_layer=vit_ckpt_layer,
            prompt=prompt,
            threshold=threshold,
            delete_tag_index=delete_tag_index,
            clip_pretraind=clip_pretraind,
            tag_list=tag_list,
            clip_version=clip_version,
            tag_list_chinese=tag_list_chinese,
            q2l_config=q2l_config,
            ram_class_threshold_path=ram_class_threshold_path,
            stage=stage)

        self.label_embed = self.create_parameter(
            shape=(self.num_class * 51, 512), default_initializer=Constant())
        self.reweight_scale = self.create_parameter(
            shape=(1, ), default_initializer=Constant(1. * np.log(1 / 0.07)))
        self.text_alignment_loss_function = AsymmetricLoss(
            gamma_neg=4, gamma_pos=0, clip=0.05)
    
    def forward(self, image_ram, caption, image_tag, parse_tag,
                imageclip=None):
        """
        call function as forward

        Args:
            image_ram: type: paddle.Tensor  shape: batch_size * 3 * 384 * 384
            caption: type: list[string]  len: batch_size
            image_tag: type: paddle.Tensor   shape: batch * class_num (e.g. 3429)   value: positive sample is 1.0, negative sample is 0.0
            parse_tag: text for image_tag
            imageclip = image for clip encoder

        Returns:
            loss: type: paddle.Tensor
        """
        assert self.stage == 'train'
        clip_feature = self.CLIP.encode_image(imageclip)
        batch_text_embed = build_text_embed(self.CLIP, caption)
        image_embeds = self.image_proj(self.visual_encoder(image_ram))
        image_atts = paddle.ones(image_embeds.shape[:-1], dtype=paddle.int32)
        ##================= Distillation from CLIP ================##
        image_cls_embeds = image_embeds[:, 0, :]
        image_spatial_embeds = image_embeds[:, 1:, :]

        loss_dis = F.l1_loss(image_cls_embeds, clip_feature)

        ##================= Image Tagging ================##
        bs = paddle.shape(image_embeds)[0]
        des_per_class = int(self.label_embed.shape[0] / self.num_class)
        image_cls_embeds = image_cls_embeds / image_cls_embeds.norm(
            axis=-1, keepdim=True)
        reweight_scale = self.reweight_scale.exp()
        logits_per_image = (reweight_scale * image_cls_embeds
                            @self.label_embed.t())
        logits_per_image = logits_per_image.reshape([bs, -1, des_per_class])
        weight_normalized = nn.functional.softmax(logits_per_image, axis=2)
        label_embed_reweight = paddle.empty([bs, self.num_class, 512])
        for i in range(bs):
            reshaped_value = self.label_embed.reshape([-1, des_per_class, 512])
            product = weight_normalized[i].unsqueeze(-1) * reshaped_value
            label_embed_reweight[i] = product.sum(axis=1)

        label_embed = nn.functional.relu(
            self.wordvec_proj(label_embed_reweight))

        tagging_embed = self.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging', )

        logits = self.fc(tagging_embed[0]).squeeze(-1)
        loss_tag = self.tagging_loss_function(logits, image_tag)

        ##================= Image-text Alignment ================##

        batch_text_embed = F.relu(
            self.wordvec_proj(
                batch_text_embed.astype(self.label_embed.dtype)))
        batch_text_embed = batch_text_embed.unsqueeze(0).tile([bs, 1, 1])
        alignment_embedding = self.tagging_head(
            encoder_embeds=batch_text_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging', )
        alignment_logits = self.fc(alignment_embedding[0]).squeeze(-1)

        with paddle.no_grad():
            alignment_targets = paddle.zeros(alignment_logits.shape)
            alignment_targets.fill_diagonal_(1)

        loss_alignment = self.text_alignment_loss_function(alignment_logits,
                                                           alignment_targets)

        return loss_tag, loss_dis, loss_alignment

    # to support paddle framework
    def inference(self, image):

        image_embeds = self.image_proj(self.visual_encoder(image))
        image_atts = paddle.ones(image_embeds.shape[:-1], dtype=paddle.int32)

        image_cls_embeds = image_embeds[:, 0, :]
        image_spatial_embeds = image_embeds[:, 1:, :]

        bs = image_spatial_embeds.shape[0]

        des_per_class = int(self.label_embed.shape[0] / self.num_class)

        image_cls_embeds = image_cls_embeds / image_cls_embeds.norm(
            axis=-1, keepdim=True)
        reweight_scale = self.reweight_scale.exp()
        logits_per_image = (reweight_scale * image_cls_embeds
                            @self.label_embed.t())
        logits_per_image = logits_per_image.reshape([bs, -1, des_per_class])

        weight_normalized = F.softmax(logits_per_image, axis=2)
        label_embed_reweight = paddle.empty([bs, self.num_class, 512])

        for i in range(bs):
            # boardingcast 
            reshaped_value = self.label_embed.reshape([-1, des_per_class, 512])
            product = weight_normalized[i].unsqueeze(-1) * reshaped_value
            label_embed_reweight[i] = product.sum(axis=1)

        label_embed = F.relu(self.wordvec_proj(label_embed_reweight))

        # recognized image tags using alignment decoder
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

        image_embeds = self.image_proj(self.visual_encoder(image))
        image_atts = paddle.ones(image_embeds.shape[:-1], dtype=paddle.int32)

        image_cls_embeds = image_embeds[:, 0, :]
        image_spatial_embeds = image_embeds[:, 1:, :]

        bs = image_spatial_embeds.shape[0]

        des_per_class = int(self.label_embed.shape[0] / self.num_class)

        image_cls_embeds = image_cls_embeds / image_cls_embeds.norm(
            axis=-1, keepdim=True)
        reweight_scale = self.reweight_scale.exp()
        logits_per_image = (reweight_scale * image_cls_embeds
                            @self.label_embed.t())
        logits_per_image = logits_per_image.reshape([bs, -1, des_per_class])

        weight_normalized = F.softmax(logits_per_image, axis=2)
        label_embed_reweight = paddle.empty([bs, self.num_class, 512])

        for i in range(bs):
            # boardingcast
            reshaped_value = self.label_embed.reshape([-1, des_per_class, 512])
            product = weight_normalized[i].unsqueeze(-1) * reshaped_value
            label_embed_reweight[i] = product.sum(axis=1)

        label_embed = F.relu(self.wordvec_proj(label_embed_reweight))

        # recognized image tags using alignment decoder
        tagging_embed = self.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging', )

        logits = self.fc(tagging_embed[0]).squeeze(-1)

        return logits


# load RAM pretrained model parameters
def ram_plus(pretrained='', **kwargs):
    model = RAM_plus(pretrained='', **kwargs)
    return model
