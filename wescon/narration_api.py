
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
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
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
from typing import Dict, Optional, Union
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from wescon.utils.common import IGNORE_ID
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
import json
from hyperpyyaml import load_hyperpyyaml
import numpy as np
from wescon.cli.frontend import CosyVoiceFrontEnd, CosyVoiceTextFrontEnd
from wescon.llm.llm import Qwen2Encoder, Qwen2LM, Qwen2ForCausalLM, make_pad_mask
from wescon.llm.qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2PreTrainedModel, Cache, DynamicCache, _prepare_4d_causal_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask, BaseModelOutputWithPast, Qwen2NAREncoder
import threading
import time
from wescon.utils.common import ras_sampling
from copy import deepcopy
from fastapi import FastAPI
from pydantic import BaseModel
import traceback
import torch
import torch.nn.functional as F
import random
from g2p_en import G2p
import torchaudio
import re
import librosa as lib
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def clean_last_char(s):
    if not s:
        return s
    # 判断最后一个字符是不是汉字或英文字母
    if re.match(r'[\u4e00-\u9fa5a-zA-Z]$', s[-1]):
        return s
    else:
        return s[:-1]

def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def contains_english(text):
    return bool(re.search(r'[a-zA-Z]', text))

def check_target_ratio(tp_tokens, end_idx, target_token_list):
    """
    统计 tp_tokens[:end_idx] 中 target token 的占比，超过50%返回 None，否则返回 tp_tokens[:end_idx]
    """
    target_set = set(target_token_list)
    segment = tp_tokens[:end_idx]
    total_count = len(segment)
    if total_count == 0:
        return None  # 空直接返回 None

    # 统计 target token 数量
    target_count = sum(1 for token in segment if token.item() in target_set)

    # 判断占比
    if target_count / total_count > 0.7:
        return None
    else:
        return segment

def argmax_with_block_penalty(logits, target_token_id, repeat_threshold=20, penalty_scale=0.5):
    """
    logits: (B, T, n_class)
    - target_token_id: 要处理的token
    - repeat_threshold: 连续块超过多少次开始惩罚
    - penalty_scale: 惩罚比例，结合原logit值
    """
    B, T, n_class = logits.size()
    pred_tokens = torch.argmax(logits, dim=-1)  # (B, T)
    penalty_mask = torch.zeros_like(logits)

    for b in range(B):
        count = 0
        start_idx = None
        for t in range(T):
            if pred_tokens[b, t] == target_token_id:
                if start_idx is None:
                    start_idx = t
                count += 1
            else:
                # 当前块结束，判断是否需要惩罚
                if count >= repeat_threshold:
                    for i, t_idx in enumerate(range(start_idx, start_idx + count)):
                        decay = (count - i) / count  # 块内越靠前惩罚越大
                        base_score = logits[b, t_idx, target_token_id].item()
                        dynamic_penalty = penalty_scale * base_score * decay
                        penalty_mask[b, t_idx, target_token_id] = dynamic_penalty
                # reset
                count = 0
                start_idx = None
        # 如果序列最后一块刚好结束在最后
        if count >= repeat_threshold:
            for i, t_idx in enumerate(range(start_idx, start_idx + count)):
                decay = (count - i) / count
                base_score = logits[b, t_idx, target_token_id].item()
                dynamic_penalty = penalty_scale * base_score * decay
                penalty_mask[b, t_idx, target_token_id] = dynamic_penalty

    # 施加惩罚
    adjusted_logits = logits - penalty_mask
    final_pred = torch.argmax(adjusted_logits, dim=-1)
    return final_pred

class BatchNormConv1d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding,
                 activation=None):
        super().__init__()
        self.conv1d = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)
        self.activation = activation

    def forward(self, x):
        x = self.conv1d(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.bn(x)

class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, ker, text_class):
        super().__init__()
        self.convs = nn.ModuleList([
            BatchNormConv1d(input_dim, hidden_dim, ker, 1, ker//2),
            BatchNormConv1d(hidden_dim, hidden_dim, ker, 1, ker//2),
        ])
        self.text_fc_out = nn.Linear(hidden_dim, text_class)
        self.bd_fc_out = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        # x: B, T, C
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)  # 卷积不处理 mask，先算
        text_out = self.text_fc_out(x.transpose(1, 2))  # [B, T, out_class]
        bd_out = self.bd_fc_out(x.transpose(1, 2))  # [B, T, out_class]
        return text_out, bd_out

@dataclass
class Wescon1stConfig(FairseqDataclass):
    text_encoder_input_size: int = field(default=512)
    llm_input_size: int = field(default=896)
    llm_output_size: int = field(default=896)
    text_token_size: int = field(default=51866)
    speech_token_size: int = field(default=6561)
    length_normalized_loss: bool = field(default=True)
    lsm_weight: float = field(default=0.0)
    spk_embed_dim: int = field(default=192)
    
    lora_rank: int = field(default=-1)
    aligner_layer: int = field(default=5)
    aligner_convdim: int = field(default=512)
    aligner_convker: int = field(default=5)
    
    partial_train: bool = field(
        default=False
    )
    partial_layers: str = field(
        default=""
    )
    qwen_pretrained_path: str = field(
        default=""
    )


class TPModule(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._attn_implementation = config._attn_implementation
        
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        position_ids=None
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, seq_length, _ = inputs_embeds.shape
        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(
                    past_key_values
                )
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds
        next_decoder_cache = None

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        hidden_states = self.norm(hidden_states)
        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=(),
            attentions=(),
        )

class Wescon1st(BaseFairseqModel):
    def __init__(self, cfg:Wescon1stConfig):
        super().__init__()
        self.text_frontend = CosyVoiceTextFrontEnd()
        self.text_frontend.add_special_tokens({'additional_special_tokens':['<|silence|>', '<|nexttext|>']})
        self.silence_token = self.text_frontend._extract_text_token('<|silence|>')[0]
        self.next_token = self.text_frontend._extract_text_token('<|nexttext|>')[0]
        self.text_pad_idx = self.text_frontend._extract_text_token('<|endoftext|>')[0]

        self.partial_train = cfg.partial_train
        self.cfg = cfg
        self.llm_input_size = self.cfg.llm_input_size
        self.speech_token_size = self.cfg.speech_token_size
        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.fill_token = 2
        self.llm_embedding = torch.nn.Embedding(2, self.cfg.llm_input_size)
        if not os.path.exists(cfg.qwen_pretrained_path):
            cfg.qwen_pretrained_path = os.environ.get("COSYVOICE2HOME") + "/CosyVoice-BlankEN"
        self.llm = Qwen2Encoder(
            cfg.qwen_pretrained_path, # init
        )
        self.llm_decoder = nn.Linear(self.cfg.llm_output_size, self.cfg.speech_token_size + 3)
        
        self.text_num = 151665
        # aligner_config = deepcopy(self.llm.model.preset_config)
        # aligner_config._name_or_path = ""
        # aligner_config.num_hidden_layers = cfg.aligner_layer
        # self.aligner = Qwen2NAREncoder(
        #     aligner_config,
        # )
        # self.aligner_decoder = Predictor(
        #     self.cfg.llm_output_size, 
        #     self.cfg.aligner_convdim,
        #     self.cfg.aligner_convker,
        #     self.text_num
        # )
        self.speech_embedding = torch.nn.Embedding(self.cfg.speech_token_size + 3, self.cfg.llm_input_size)
        self.hift = None
        self.flow = None
        self.sample_rate = 24000
        # infer
        self.mel_cache_len = 20

    def pad_unpad_sequence(self, sos_eos_emb, text_token, text_token_len, task_id_emb, speech_token, speech_token_len):
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        lm_input = [torch.concat([sos_eos_emb.squeeze(dim=0), text_token[i], task_id_emb.squeeze(dim=0), speech_token[i]], dim=0) for i in range(len(text_token))]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
    ):
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (self.speech_token_size not in top_ids):
                break
        return top_ids
    
    def sampling_tokens(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
    ):
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (self.speech_token_size not in top_ids):
                break
        return top_ids
    
    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
            a_h_n_sa_su_text_lens=None,
            a_h_n_sa_su_st_lens=None,
            text_control=(-1, -1),
            last_item=False,
            lang=""
    ):
        # start = time.perf_counter()
        tgt_text_len = text.size(1)
        prompt_speech_len = prompt_speech_token.size(1)
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.llm.model.model.embed_tokens(text)

        # 2. encode embedding
        embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(
                1, 0, self.llm_input_size, dtype=text.dtype
            ).to(device)
        lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * 0)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        cache = None
        cache_align = None
        y_preds = []
        # end = time.perf_counter()
        # print(f"infer内部进入循环前耗时: {end - start:.6f} 秒")
        # start = time.perf_counter()
        for i in range(max_len):
            temp_mask = torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool)
            y_pred, cache = self.llm.forward_one_step(lm_input,
                                                    masks=temp_mask,
                                                    cache=cache,
                                                    a_h_n_sa_su_text_lens=a_h_n_sa_su_text_lens,
                                                    a_h_n_sa_su_st_lens=a_h_n_sa_su_st_lens,
                                                    text_control=text_control)
            # y_preds.append(y_pred)
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            # print(f"max_len:{max_len}, y_pred:{top_ids}")
            if top_ids == self.speech_token_size:
                break
            if top_ids > self.speech_token_size:
                continue
            # in stream mode, yield token one by one
            # yield top_ids
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
            
        return out_tokens
    
    def init_infer_modules(self, 
                        device,
                        model_dir="", 
                        instruct=True,
                        fm_model=None):
        model_dir = os.environ.get("COSYVOICE2HOME", "model_cache/CosyVoice2-0.5B")
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f)
        print(configs)
        self.device = device
        self.frontend = CosyVoiceFrontEnd('{}/campplus.onnx'.format(model_dir),
                                        '{}/speech_tokenizer_v2.onnx'.format(model_dir),
                                        '{}/spk2info.pt'.format(model_dir),
                                        instruct,
                                        configs['allowed_special'],
                                        text_frontend=self.text_frontend,
                                        device=device)
        if fm_model is not None:
            self.flow = fm_model
            self.flow.to(device).eval()
            self.our_fm = True
        else:
            self.flow = configs["flow"]
            flow_state_dict = {k.replace('generator.', ''): v for k, v in torch.load('{}/flow.pt'.format(model_dir), map_location=self.device).items()}
            self.flow.load_state_dict(flow_state_dict, strict=True)
            self.flow.to(device).eval()
            self.our_fm = False
        
        self.hift = configs["hift"]
        self.hift.load_state_dict(torch.load('{}/hift.pt'.format(model_dir), map_location=device), strict=True)
        self.hift.to(device).eval()
        self.eval()
        self.device = device
        self.model_dir = model_dir
        self.lock = threading.Lock()
        del configs
        
        # 
        self.token_min_hop_len = 2 * self.flow.input_frame_rate
        self.token_max_hop_len = 4 * self.flow.input_frame_rate
        self.token_overlap_len = 20
        # here we fix set flow.decoder.estimator.static_chunk_size = 0 for compatibability
        if self.our_fm:
            self.flow.causal_masked_diff.decoder.estimator.static_chunk_size = 0
            self.flow.causal_masked_diff.decoder.fp16 = False
        else:
            self.flow.decoder.estimator.static_chunk_size = 0
            self.flow.decoder.fp16 = False
        # mel fade in out
        self.mel_overlap_len = int(self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = 20
        self.source_cache_len = int(self.mel_cache_len * 256)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.stream_scale_factor = 1
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'
        # self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.mel_overlap_dict = {}
        self.flow_cache_dict = {}
        self.hift_cache_dict = {}
        self.sampling = ras_sampling
    
    def inference_whole(self, 
                        text, 
                        flow_embedding, 
                        llm_embedding=torch.zeros(0, 192),
                        prompt_text=torch.zeros(1, 0, dtype=torch.int32),
                        llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
                        flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
                        prompt_speech_feat=torch.zeros(1, 0, 80), speed=1.0, 
                        prompt_sp = 1,
                        return_speech = False,
                        **kwargs):
        if prompt_sp != 1.0:
            llm_prompt_speech_token = resample_by_stride(llm_prompt_speech_token[0], prompt_sp).unsqueeze(0)
        this_tts_speech_token, (last_text, last_st) = self.inference(text=text.to(self.device),
                        text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                        prompt_text=prompt_text.to(self.device),
                        prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                        prompt_speech_token=llm_prompt_speech_token.to(self.device),
                        prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                        embedding=llm_embedding.to(self.device))
        if return_speech:
            this_tts_speech_token = torch.tensor(this_tts_speech_token).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                            prompt_token=flow_prompt_speech_token,
                                            prompt_feat=prompt_speech_feat,
                                            embedding=flow_embedding,
                                            token_offset=0,
                                            finalize=True,
                                            speed=speed).cpu()
        else:
            this_tts_speech = None
        return this_tts_speech_token, this_tts_speech
    
    def token2wav(self, token, prompt_token, prompt_feat, embedding, token_offset, finalize=False, speed=1.0):
        # print(token)
        tts_mel, _ = self.flow.inference(token=token.to(self.device),
                                        token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                        prompt_token=prompt_token.to(self.device),
                                        prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                        prompt_feat=prompt_feat.to(self.device),
                                        prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                        embedding=embedding.to(self.device),
                                        finalize=finalize)
        tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
        # append hift cache
        hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if speed != 1.0:
            tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
        tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
        return tts_speech
    
    def generate_speech(self,
                            this_tts_speech_token,
                            flow_prompt_speech_token,
                            prompt_speech_feat,
                            flow_embedding,
                            speed=1.0,
                            **kwargs):
        this_tts_speech_token = torch.tensor(this_tts_speech_token).unsqueeze(dim=0)
        this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                        prompt_token=flow_prompt_speech_token,
                                        prompt_feat=prompt_speech_feat,
                                        embedding=flow_embedding,
                                        token_offset=0,
                                        finalize=True,
                                        speed=speed).cpu()
        return this_tts_speech
            
    def inference_st(self, 
                        text, 
                        llm_embedding=torch.zeros(0, 192),
                        prompt_text=torch.zeros(1, 0, dtype=torch.int32),
                        llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
                        prompt_sp=1,
                        last_item=False,
                        lang="",
                        **kwargs):
        this_tts_speech_token = self.inference(text=text.to(self.device),
                        text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                        prompt_text=prompt_text.to(self.device),
                        prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                        prompt_speech_token=llm_prompt_speech_token.to(self.device),
                        prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                        embedding=llm_embedding.to(self.device),
                        last_item=last_item,
                        lang=lang)
        return torch.tensor(this_tts_speech_token).to(torch.long)
    
def resample_by_stride(tensor, scale):
    length = len(tensor)
    new_length = max(1, int(length * scale))  # 计算新的长度
    indices = torch.linspace(0, length - 1, new_length).round().long()  # 均匀采样索引
    return tensor[indices]

def make_encoder_attention_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Generate attention mask for torch.nn.functional.scaled_dot_product_attention.
    Masked positions are -inf, valid positions are 0.0 (float type).

    Args:
        lengths (torch.Tensor): Tensor of shape [B], each element is the valid length of the sequence.
        max_len (int): Optional maximum sequence length. If 0, use max(lengths).

    Returns:
        torch.Tensor: Attention mask of shape [B, 1, T, T].
    """
    NEG_INF = -65500
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()

    # Step 1: Create padding mask [B, T], True means PAD
    seq_range = torch.arange(0, max_len, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    pad_mask = seq_range_expand >= seq_length_expand  # [B, T], True is PAD

    # Step 2: Expand to attention mask shape [B, 1, T, T]
    attn_mask = pad_mask.unsqueeze(1).unsqueeze(2).expand(-1, 1, max_len, -1)  # mask key

    # Step 3: Convert to float mask: PAD -> -inf, valid -> 0.0
    attn_mask = attn_mask.masked_fill(attn_mask, NEG_INF).masked_fill(~attn_mask, 0.0)

    return ~attn_mask  # [B, 1, T, T]


def extract_with_targetset_limit(tensor, target_list=[151663, 151664], max_sil=10, type_num=1, max_len=30):
    """
    向前回溯 type_num 种非 target_list 内 token，确保返回结果不带前置 target
    target_list 支持多个 target token
    """
    tot_len = len(tensor)
    target_set = set(target_list)

    # Step 1: 定位最后的非 target
    mask = ~torch.isin(tensor, torch.tensor(target_list, device=tensor.device))
    non_target_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if non_target_indices.numel() == 0:
        return None, None, tot_len  # 全是 target

    last_non_target_idx = non_target_indices[-1].item()

    # Step 2: 回溯收集 type_num 种 non-target token
    unique_tokens = set()
    non_target_pos = []
    idx = last_non_target_idx
    while idx >= 0:
        token = tensor[idx].item()
        if token not in target_set:
            if token not in unique_tokens:
                unique_tokens.add(token)
            non_target_pos.append(idx)
            if len(unique_tokens) == type_num:
                # 将连续相同 token 全部纳入
                token_type = token
                while idx > 0 and tensor[idx - 1].item() == token_type:
                    idx -= 1
                    non_target_pos.append(idx)
                break
        idx -= 1

    if not non_target_pos:
        return None, None, tot_len  # 没找到足够的非 target

    start_idx = min(non_target_pos)

    # Step 3: 后面拼接 target（允许拼接 max_sil 个 target）
    target_count = 0
    end_idx = last_non_target_idx + 1
    while end_idx < len(tensor) and tensor[end_idx].item() in target_set and target_count < max_sil:
        end_idx += 1
        target_count += 1

    # Step 4: 限制 max_len
    if end_idx - start_idx > max_len:
        start_idx = end_idx - max_len

    return tensor[start_idx:end_idx], start_idx, end_idx

def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav)
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech

def load_pretrained_tp_models(model, pretrained_checkpoint):
    state = torch.load(pretrained_checkpoint)["model"]
    model_state = model.state_dict()
    cur_model_keys = deepcopy(list(model_state.keys()))
    pretrained_dict = {}
    for k, v in state.items():
        if k in model_state.keys():
            if v.size() == model_state[k].size():
                pretrained_dict[k] = v.to(model_state[k].dtype)
                cur_model_keys.remove(k)
            else:
                pretrained_dict[k] = model_state[k].to(model_state[k].dtype)
                print(k, v.size(), model_state[k].size(), "size error")
    print("loaded tp: "+str(pretrained_dict.keys()))
    print("not loaded tp: "+str(cur_model_keys))
    model_state.update(pretrained_dict)
    model.load_state_dict(model_state)
    return model

########################################################################################
def load_spk_info(model, prompt_speech):
    global_prompt_wav = load_wav(prompt_speech, target_sr=16000)
    global_prompt = model.frontend.speaker_infos(global_prompt_wav, model.sample_rate)
    return global_prompt, global_prompt_wav

def rms_normalize(waveform: torch.Tensor, target_rms: float = 0.05) -> torch.Tensor:
    """
    Normalize waveform to a target RMS level.
    waveform: (1, T) or (T,) torch.Tensor
    target_rms: target root mean square energy
    """
    if waveform.ndim == 2:
        waveform = waveform.squeeze(0)
    
    rms = torch.sqrt(torch.mean(waveform ** 2) + 1e-8)
    gain = target_rms / rms
    return torch.clamp(waveform * gain, -1.0, 1.0).unsqueeze(0)

app = FastAPI()

class ArrayInput(BaseModel):
    text: list  
    emotion: list 
    duration: list  
    global_emotion: str
    speaker: str
    wer_threshold: float
    ssim_threshold: float
    esim_threshold: float
    languages: list

##########################################################

device = torch.device("cuda:0")
g2p_model = G2p()
config = Wescon1stConfig()
model = Wescon1st(config)
ckpt = r"model_cache/narrations/man1/checkpoint_last.pt"
model = load_pretrained_tp_models(model, ckpt).to(device)
model.init_infer_modules(device)
prompt_speech = r"model_cache/narrations/man1/1064_part.wav"
# prompt_trans = r"晚上做噩梦怎么办呢？算了我还是别见了。老头呢很懊恼，退朝之后把这件事儿跟马骥就说了。马骥说，没关系没关系，我在您这儿呆的也挺开心的。马骥呢在老头这儿就又住了几天。"
prompt_trans = r"晚上做噩梦怎么办呢，算了我还是别见了，老头呢很懊恼。"

@app.post("/tts_narration")
def emotional_tts_genshin(data: ArrayInput):
    global_emotion = data.global_emotion
    speaker = data.speaker
    generated_speech_tokens = []
    generated_emo_speech = []

    global prompt_trans
    prompt_trans_normed = model.frontend.text_normalize(prompt_trans, split=False, text_frontend=True)
    prompt_text_token, prompt_text_token_len = model.frontend._extract_text_token(prompt_trans_normed)
    global_prompt, global_prompt_wav = load_spk_info(model, prompt_speech)
    prompt_speech_token, prompt_speech_token_len = model.frontend._extract_speech_token(global_prompt_wav)
    
    for idx, (text, emotion, duration, lang) in enumerate(zip(
        data.text, data.emotion, data.duration, data.languages
    )):
        generated_ok = False
        regenerate_time = 10
        all_eval_results = []
        while not generated_ok and regenerate_time > 0:
            try:
                temp_line_speech_token = []
                for text_item in model.frontend.text_normalize(text, split=True):
                    model_input = {
                        "text": model.frontend._extract_text_token(text_item)[0],
                        "prompt_text": prompt_text_token,
                        "llm_prompt_speech_token": prompt_speech_token,
                    }
                        
                    for key in model_input.keys():
                        try:
                            model_input[key] = model_input[key].to(device)
                            # print(f"key:{key};size:{str(model_input[key].size())}")
                        except:
                            pass
                    print(text_item)
                    speech_token = model.inference_st(**model_input)
                    temp_line_speech_token.append(speech_token)
                temp_line_speech_token = torch.cat(temp_line_speech_token, dim=0)
                print(temp_line_speech_token.size())
                speech = model.generate_speech(temp_line_speech_token, **global_prompt)
            except Exception as e:
                traceback.print_exc()
                print("retry")
                continue
            eval_info = {
                "gen_array": [speech.flatten().numpy().tolist()],
                "emotion_ref": [global_prompt_wav.flatten().numpy().tolist()],
                "tgt_text": [text],
                "speaker_ref": global_prompt_wav.flatten().numpy().tolist(),
                "languages": [lang]
            }
            response = requests.post("http://127.0.0.1:8100/predict", json=eval_info, proxies={"http": None, "https": None})
            print("Raw Text:", response.text) 
            eval_result = json.loads(response.text)
            all_eval_results.append({
                "speech_token": temp_line_speech_token,
                "wer": np.mean(eval_result["wers"]),
                "s_sim": np.mean(eval_result["s_sims"]),
                "e_sim": np.mean(eval_result["e_sims"]),
                "emo_speech": speech.flatten().numpy()
            })
            if np.mean(eval_result["wers"]) > data.wer_threshold:
                regenerate_time -= 1
                continue
            if np.mean(eval_result["s_sims"]) < data.ssim_threshold:
                regenerate_time -= 1
                continue
            if np.mean(eval_result["e_sims"]) < data.esim_threshold:
                regenerate_time -= 1
                continue
            generated_speech_tokens.append(temp_line_speech_token)
            generated_ok = True
            generated_emo_speech.append(all_eval_results[-1]["emo_speech"])

        if regenerate_time <= 0 and not generated_ok:
            # 多指标排序：WER最小，s_sim最大，e_sim最大
            all_eval_results.sort(key=lambda x: (x["wer"], -x["s_sim"], -x["e_sim"]))
            best = all_eval_results[0]
            generated_speech_tokens.append(best["speech_token"])
            generated_ok = True
            generated_emo_speech.append(best["emo_speech"])

    temp_line_speech_token = torch.cat(generated_speech_tokens, dim=0)
    speech = model.generate_speech(temp_line_speech_token, **global_prompt)
    speech = rms_normalize(speech)
    return {"speech": speech.flatten().numpy().tolist(), "sample_rate": model.sample_rate, "generated_emo_speech": np.concatenate(generated_emo_speech).tolist()}

########################################################################################


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("wescon.narration_api:app", host="0.0.0.0", port=8103, reload=False)