# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time: 2023/12/27 17:01
@Author: shiqixin.set
@File: model_util.py
@Software: PyCharm
@desc:
"""

import base64
import struct
import torch
from argparse import Namespace
from transformers import AutoModel, AutoTokenizer
import random
import os

# from util.text_util import is_chinese_or_english
# import tiktoken

# client = OpenAI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_luotuo_model = None

_luotuo_model_en = None
_luotuo_en_tokenizer = None

_enc_model = None


def float_array_to_base64(float_arr):
    """
    float转base64
    :param float_arr:
    :return:
    """
    byte_array = b''

    for f in float_arr:
        # 将每个浮点数打包为4字节
        num_bytes = struct.pack('!f', f)
        byte_array += num_bytes

    # 将字节数组进行base64编码
    base64_data = base64.b64encode(byte_array)

    return base64_data.decode('utf-8')


def base64_to_float_array(base64_data):
    """
    base64转float
    :param base64_data:
    :return:
    """
    byte_array = base64.b64decode(base64_data)

    float_array = []

    # 每 4 个字节解析为一个浮点数
    for i in range(0, len(byte_array), 4):
        num = struct.unpack('!f', byte_array[i:i + 4])[0]
        float_array.append(num)

    return float_array


def download_luotuo_models():
    """
    下载luotuo模型
    :return:
    """
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "luotuo_bert_model")
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
    if len(os.listdir(model_path)) == 0:
        print("正在下载Luotuo-Bert")
        # Import our models. The package will take care of downloading the models automatically
        model_args = Namespace(do_mlm=None, pooler_type="cls", temp=0.05, mlp_only_train=False,
                               init_embeddings_model=None)
        model = AutoModel.from_pretrained("silk-road/luotuo-bert-medium", trust_remote_code=True,
                                          model_args=model_args).to(
            device)
        model.save_pretrained(model_path)
        print("Luotuo-Bert下载完毕")
        return model
    else:
        print("正在加载Luotuo-Bert")
        model_args = Namespace(do_mlm=None, pooler_type="cls", temp=0.05, mlp_only_train=False,
                               init_embeddings_model=None)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, model_args=model_args).to(
            device)
        print("Luotuo-Bert加载完毕")
        return model


def get_luotuo_model():
    global _luotuo_model
    if _luotuo_model is None:
        _luotuo_model = download_luotuo_models()
    return _luotuo_model


def luotuo_embedding(model, texts):
    """
    用骆驼模型生成embedding
    :param model:
    :param texts:
    :return:
    """
    # Tokenize the texts_source
    tokenizer = AutoTokenizer.from_pretrained("silk-road/luotuo-bert-medium")
    inputs = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
    inputs = inputs.to(device)
    # Extract the embeddings
    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True, sent_emb=True).pooler_output
    return embeddings


def get_embedding_for_chinese(model, texts):
    """
    获取embedding
    :param model:
    :param texts:
    :return:
    """
    model = model.to(device)
    # str or strList
    texts = texts if isinstance(texts, list) else [texts]
    # 截断
    for i in range(len(texts)):
        if len(texts[i]) > 510:
            texts[i] = texts[i][:510]
    if len(texts) >= 64:
        embeddings = []
        chunk_size = 64
        for i in range(0, len(texts), chunk_size):
            embeddings.append(luotuo_embedding(model, texts[i: i + chunk_size]))
        return torch.cat(embeddings, dim=0)
    else:
        return luotuo_embedding(model, texts)


def get_embedding_for_english(text, model="text-embedding-ada-002"):
    """
    获取英文embedding
    :param text:
    :param model:
    :return:
    """
    # #todo 使用luotuo-bert-en
    # text = text.replace("\n", " ")
    # return client.embeddings.create(input=[text], model=model).data[0].embedding
    return


def luotuo_openai_embedding(texts, is_chinese=None):
    """
        when input is chinese, use luotuo_embedding
        when input is english, use openai_embedding
        texts can be a list or a string
        when texts is a list, return a list of embeddings, using batch inference
        when texts is a string, return a single embedding
    """
    if isinstance(texts, list):
        return [embed.cpu().tolist() for embed in get_embedding_for_chinese(get_luotuo_model(), texts)]
    else:
        return get_embedding_for_chinese(get_luotuo_model(), texts)[0].cpu().tolist()

    # openai_key = os.environ.get("OPENAI_API_KEY")
    #
    # if isinstance(texts, list):
    #     index = random.randint(0, len(texts) - 1)
    #     if openai_key is None or is_chinese_or_english(texts[index]) == "chinese":
    #         return [embed.cpu().tolist() for embed in get_embedding_for_chinese(get_luotuo_model(), texts)]
    #     else:
    #         return [get_embedding_for_english(text) for text in texts]
    # else:
    #     if openai_key is None or is_chinese_or_english(texts) == "chinese":
    #         return get_embedding_for_chinese(get_luotuo_model(), texts)[0].cpu().tolist()
    #     else:
    #         return get_embedding_for_english(texts)

# def tiktokenizer(text):
#     """
#     分词器计算
#     :param text:
#     :return:
#     """
#     global _enc_model
#
#     if _enc_model is None:
#         _enc_model = tiktoken.get_encoding("cl100k_base")
#
#     return len(_enc_model.encode(text))
