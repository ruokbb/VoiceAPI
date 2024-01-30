# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time: 2024/1/29 17:47
@Author: shiqixin.set
@File: model_loading.py
@Software: PyCharm
@desc: 
"""

import torch
import os
from GPT_SoVITS.feature_extractor import cnhubert
from transformers import AutoModelForMaskedLM, AutoTokenizer
from GPT_SoVITS.module.models import SynthesizerTrn
from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule

class DictToAttrRecursive:
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            if isinstance(value, dict):
                # 如果值是字典，递归调用构造函数
                setattr(self, key, DictToAttrRecursive(value))
            else:
                setattr(self, key, value)

is_half = eval(os.environ.get("is_half","True"))

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    if (
            ("16" in gpu_name and "V100" not in gpu_name.upper())
            or "P40" in gpu_name.upper()
            or "P10" in gpu_name.upper()
            or "1060" in gpu_name
            or "1070" in gpu_name
            or "1080" in gpu_name
    ):
        is_half=False
if(device== "cpu"):is_half=False
print("半精：{}".format(is_half))

cnhubert_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
cnhubert.cnhubert_base_path = cnhubert_path
ssl_model = cnhubert.get_model()
if is_half:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)

dict_language = {
    "中文": "zh",
    "英文": "en",
    "日文": "ja",
    "ZH": "zh",
    "EN": "en",
    "JA": "ja",
    "zh": "zh",
    "en": "en",
    "ja": "ja"
}

bert_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)

hz = 50

# 不同模型配置
default_sovits_path = "GPT_SoVITS/pretrained_models/s2G488k.pth"
default_dict_s2 = torch.load(default_sovits_path, map_location="cpu")
default_hps = default_dict_s2["config"]
default_hps = DictToAttrRecursive(default_hps)
default_hps.model.semantic_frame_rate = "25hz"

default_gpt_path = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
default_dict_s1 = torch.load(default_gpt_path, map_location="cpu")
default_config = default_dict_s1["config"]
default_max_sec = default_config['data']['max_sec']

default_t2s_model = Text2SemanticLightningModule(default_config, "****", is_train=False)
default_t2s_model.load_state_dict(default_dict_s1["weight"])
if is_half:
    default_t2s_model = default_t2s_model.half()
default_t2s_model = default_t2s_model.to(device)
default_t2s_model.eval()

default_vq_model = SynthesizerTrn(
    default_hps.data.filter_length // 2 + 1,
    default_hps.train.segment_size // default_hps.data.hop_length,
    n_speakers=default_hps.data.n_speakers,
    **default_hps.model)
if is_half:
    default_vq_model = default_vq_model.half().to(device)
else:
    default_vq_model = default_vq_model.to(device)
default_vq_model.eval()
print(default_vq_model.load_state_dict(default_dict_s2["weight"], strict=False))


# suiji
suiji_sovits_path = "model/suiji/suiji_e25_s1525.pth"
suiji_dict_s2 = torch.load(suiji_sovits_path, map_location="cpu")
suiji_hps = suiji_dict_s2["config"]
suiji_hps = DictToAttrRecursive(suiji_hps)
suiji_hps.model.semantic_frame_rate = "25hz"

suiji_gpt_path = "model/suiji/suiji-e5.ckpt"
suiji_dict_s1 = torch.load(suiji_gpt_path, map_location="cpu")
suiji_config = suiji_dict_s1["config"]
suiji_max_sec = suiji_config['data']['max_sec']

suiji_t2s_model = Text2SemanticLightningModule(suiji_config, "****", is_train=False)
suiji_t2s_model.load_state_dict(suiji_dict_s1["weight"])
if is_half:
    suiji_t2s_model = suiji_t2s_model.half()
suiji_t2s_model = suiji_t2s_model.to(device)
suiji_t2s_model.eval()

suiji_vq_model = SynthesizerTrn(
    suiji_hps.data.filter_length // 2 + 1,
    suiji_hps.train.segment_size // suiji_hps.data.hop_length,
    n_speakers=suiji_hps.data.n_speakers,
    **suiji_hps.model)
if is_half:
    suiji_vq_model = suiji_vq_model.half().to(device)
else:
    suiji_vq_model = suiji_vq_model.to(device)
suiji_vq_model.eval()
print(suiji_vq_model.load_state_dict(suiji_dict_s2["weight"], strict=False))

# 模型数据
model_data = {
    "default": {
        "sovits_path": default_sovits_path,
        "gpt_path": default_gpt_path,
        "hps": default_hps,
        "vq_model":default_vq_model,
        "max_sec":default_max_sec,
        "t2s_model": default_t2s_model,
        "top_k": default_config['inference']['top_k']
    },
    "suiji": {
        "sovits_path": suiji_sovits_path,
        "gpt_path": suiji_gpt_path,
        "hps": suiji_hps,
        "vq_model": suiji_vq_model,
        "max_sec": suiji_max_sec,
        "t2s_model": suiji_t2s_model,
        "top_k": suiji_config['inference']['top_k']
    },
}