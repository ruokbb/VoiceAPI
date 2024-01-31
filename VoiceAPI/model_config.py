# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time: 2024/1/29 17:47
@Author: shiqixin.set
@File: model_config.py
@Software: PyCharm
@desc: 
"""

import torch
import os
from GPT_SoVITS.feature_extractor import cnhubert
from transformers import AutoModelForMaskedLM, AutoTokenizer
from GPT_SoVITS.module.models import SynthesizerTrn
from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from VoiceAPI.util.model_util import luotuo_openai_embedding
from VoiceAPI.ChromaDB import ChromaDB


class DictToAttrRecursive:
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            if isinstance(value, dict):
                # 如果值是字典，递归调用构造函数
                setattr(self, key, DictToAttrRecursive(value))
            else:
                setattr(self, key, value)


is_half = eval(os.environ.get("is_half", "True"))

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
        is_half = False
if (device == "cpu"): is_half = False
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
model_path_config = {
    "default": {
        "gpt": "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
        "sovits": "GPT_SoVITS/pretrained_models/s2G488k.pth",
        "slicer": "",
    },
    "suiji": {
        "gpt": "VoiceAPI/model/suiji/suiji-e5.ckpt",
        "sovits": "VoiceAPI/model/suiji/suiji_e25_s1525.pth",
        "slicer": "/home/sqx/vocal/suiji/suiji.list",
    },
}

# 填充model_data
model_data = {}
for k, v in model_path_config:
    sovits_path = v["sovits"]
    dict_s2 = torch.load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"

    gpt_path = v["gpt"]
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config['data']['max_sec']

    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()

    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    if is_half:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))

    # 加载bert
    if v["slicer"] == "":
        continue

    db = ChromaDB()
    db_folder = os.path.join(os.path.dirname(__file__), "model", k, "db")
    if os.path.exists(db_folder):
        db.load(db_folder)
    else:
        wav_data_list = []
        embedding_list = []
        with open(v["slicer"], "r", encoding="utf-8") as f:
            slicer_data = f.readlines()
            for line in slicer_data:
                text = line.split("|")[-1]
                wav_path = line.split("|")[0]
                language = line.split("|")[-2]
                embedding = luotuo_openai_embedding(text)
                wav_data_list.append(wav_path + "|" + text + "|" + language)
                embedding_list.append(embedding)
        db.init_from_docs(embedding_list, wav_data_list)
        db.save(db_folder)


    model_data[k] = {
        "hps": hps,
        "vq_model": vq_model,
        "max_sec": max_sec,
        "t2s_model": t2s_model,
        "top_k": config['inference']['top_k'],
        "db": db
    }

