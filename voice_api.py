# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time: 2024/1/30 15:32
@Author: shiqixin.set
@File: voice_api.py
@Software: PyCharm
@desc: 
"""


from fastapi import File, UploadFile, Form, Header
import json
import os
from fastapi import FastAPI, HTTPException
import uvicorn
from starlette.responses import Response
from fastapi.responses import StreamingResponse, JSONResponse
from VoiceAPI.product import get_tts_wav
import torch
import soundfile as sf
from io import BytesIO
from VoiceAPI.model_config import model_data
from VoiceAPI.util.model_util import luotuo_openai_embedding

API_DIR = "api"
USER_DATA_PATH = os.path.join(API_DIR, "user_data.json")
USER_SOURCE_DIR_PATH = os.path.join(API_DIR, "user_source")
os.makedirs(USER_SOURCE_DIR_PATH, exist_ok=True)

app = FastAPI()


def user_check(token):
    """
    判断user是否记录
    :param token:
    :return:
    """
    with open(USER_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        if token in data:
            return True
        else:
            return False


def handle(refer_wav_path, prompt_text, prompt_language, text, text_language, model_name="default"):
    if (
            refer_wav_path == "" or refer_wav_path is None
            or prompt_text == "" or prompt_text is None
            or prompt_language == "" or prompt_language is None
    ):
        if model_name == "default":
            return JSONResponse({"code": 400, "message": "未指定参考音频"}, status_code=400)
        else:
            # 从db中找
            db = model_data[model_name]["db"]
            eb = luotuo_openai_embedding(text)
            refer_data = db.search(eb, 1)
            refer_wav_path = refer_data.split("|")[0]
            prompt_text = refer_data.split("|")[1]
            prompt_language = refer_data.split("|")[2]

    with torch.no_grad():
        gen = get_tts_wav(
            refer_wav_path, prompt_text, prompt_language, text, text_language, model_name
        )
        sampling_rate, audio_data = next(gen)

    wav = BytesIO()
    sf.write(wav, audio_data, sampling_rate, format="wav")
    wav.seek(0)

    torch.cuda.empty_cache()
    try:
        torch.mps.empty_cache()
    except:
        pass
    return StreamingResponse(wav, media_type="audio/wav")


from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request


class VerifyTokenMiddleware(BaseHTTPMiddleware):
    async def dispatch(
            self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.url.path == "/add_user":
            return await call_next(request)

        token = request.headers.get("token")
        if token and user_check(token):
            if not os.path.exists(os.path.join(USER_SOURCE_DIR_PATH, token)):
                user_source_path = os.path.join(USER_SOURCE_DIR_PATH, token)
                os.makedirs(user_source_path, exist_ok=True)
                data_path = os.path.join(user_source_path, "data.json")
                if not os.path.exists(data_path):
                    with open(data_path, "w", encoding="utf-8") as f:
                        json.dump({}, f)
        else:
            raise HTTPException(status_code=403, detail="token不存在")
        return await call_next(request)


app.add_middleware(VerifyTokenMiddleware)


@app.get("/")
async def tts_endpoint(
        refer_name: str = None,
        text: str = None,
        text_language: str = "zh",
        model_name: str = "default",
        token: str = Header("default")
):
    data_path = os.path.join(USER_SOURCE_DIR_PATH, token, "data.json")
    with open(data_path, "r", encoding="UTF-8") as f:
        data = json.load(f)
        if refer_name not in data and model_name == "default":
            return JSONResponse({"code": 400, "message": "参考音频不存在，请重新上传"})
        else:
            if refer_name:
                refer_wav_path = os.path.join(USER_SOURCE_DIR_PATH, token, data[refer_name][0])
                prompt_text = data[refer_name][1]
                prompt_language = data[refer_name][2]
                return handle(refer_wav_path, prompt_text, prompt_language, text, text_language, model_name)
            else:
                return handle(None, None, "zh", text, text_language, model_name)


@app.post("/upload_refer")
async def upload_refer(
        file: UploadFile = File(...),
        file_text: str = Form(...),
        file_language: str = Form(...),
        token: str = Header("default")
):
    file_type = file.filename.split(".")[-1]
    file_type_list = ["opus", "flac", "webm", "wav", "m4a", "ogg", "oga", "mp3"]
    if file_type not in file_type_list:
        return JSONResponse({"code": 400, "message": "文件类型不符合:{}".format(file_type_list)})
    # 保存视频
    user_source_path = os.path.join(USER_SOURCE_DIR_PATH, token)
    file_path = os.path.join(user_source_path, file.filename)
    with open(file_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    # 记录文字
    user_source_data_path = os.path.join(user_source_path, "data.json")
    with open(user_source_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data[os.path.splitext(file.filename)[0]] = [file.filename, file_text, file_language]

    with open(user_source_data_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    return JSONResponse({"code": 0, "message": "Success"})


@app.post("/add_user")
async def control(request: Request):
    json_post_raw = await request.json()
    with open(USER_DATA_PATH, "r", encoding="UTF-8") as f:
        data = json.load(f)
    data[json_post_raw.get("token")] = {
        "name": json_post_raw.get("name"),
        "level": json_post_raw.get("level")
    }
    with open(USER_DATA_PATH, "w", encoding="UTF-8") as f:
        json.dump(data, f)
    return JSONResponse({"code": 0, "message": "Success"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6553, workers=1)
