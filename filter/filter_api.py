import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import numpy as np
import os
import soundfile as sf
from funasr import AutoModel
from jiwer import wer
import torchaudio
import json
import regex as re
import time
import csv
import librosa as lib
from filter.Resemblyzer.resemblyzer import VoiceEncoder, preprocess_wav
from filter.text2ipa import en2ipa, zh2ipa
from filter.SenseVoice.model import SenseVoiceSmall
import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm
import string
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import torch
from fastapi import FastAPI
from pydantic import BaseModel
import traceback
# uvicorn your_file_name:app --reload

app = FastAPI()

# ✅ 模型加载在全局（服务启动时就会加载一次）
asr_model, asr_kwargs = SenseVoiceSmall.from_pretrained(model="iic/SenseVoiceSmall", device="cuda", disable_update=False)
resemb_model = VoiceEncoder(device=torch.device("cuda"))
resemb_model.eval()
emo2vec_model = AutoModel(model="iic/emotion2vec_plus_large", disable_update=False, device="cuda")

class ArrayInput(BaseModel):
    gen_array: list  
    tgt_text: list  
    speaker_ref: list
    emotion_ref: list 
    languages: list

@app.post("/predict")
def filter_evaluate(data: ArrayInput):
    try:
        gen_array = np.array(data.gen_array) # N x T
        emotion_ref = np.array(data.emotion_ref)
        tgt_text = data.tgt_text
        wers = []
        s_sims = []
        e_sims = []
        est_txts = []
        with torch.no_grad():
            for gen_array, emotion_ref, tgt_text, lang in zip(
                np.array(data.gen_array), np.array(data.emotion_ref), 
                data.tgt_text, data.languages
            ):
                wer, est_txt = wer_metric(tgt_text, gen_array, asr_model, asr_kwargs, lang)
                est_txts.append(est_txt)
                s_sim = cos_simi_resemb(resemb_model, gen_array, np.array(data.speaker_ref))
                e_sim = emo_sim(emotion_ref, gen_array, emo2vec_model)
                wers.append(wer)
                s_sims.append(s_sim)
                e_sims.append(e_sim)
        return {
            "wers": wers, 
            "s_sims": s_sims, 
            "e_sims": e_sims,
            "est_txts": est_txts
        }
    except Exception as e:
        print("🔥 Exception caught during /predict:")
        print(traceback.format_exc())
   
SAMPLING_RATE = 16000
INPUT_LENGTH = 1

def retain_chinese_english(text):
    # 保留中文、英文、数字、常见标点（中英文），删除其他字符（emoji、特殊符号等）
    pattern = re.compile(r'[^\u4e00-\u9fffA-Za-z0-9，。！？：；、“”‘’"\'(),.?!:;\s]')
    return pattern.sub('', text)

def remove_emoji(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # 😀-😟 表情
        "\U0001F300-\U0001F5FF"  # 图形
        "\U0001F680-\U0001F6FF"  # 交通工具
        "\U0001F1E0-\U0001F1FF"  # 国旗
        "\U00002700-\U000027BF"  # 杂项符号
        "\U0001F900-\U0001F9FF"  # 补充表情
        "\U0001FA70-\U0001FAFF"  # 新 emoji
        "\U00002600-\U000026FF"  # 杂项符号
        "\U000002B00-\U00002BFF"  # 箭头
        "\U00002300-\U000023FF"  # 技术符号
        "\u200d"                 # zero width joiner
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)

def remove_punctuation_and_whitespace(text):
    return re.sub(r'[\p{P}\s]+', '', text, flags=re.UNICODE)

def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav)
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech

def normalize_text(text, remove_punctuation=False):
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\xa0]', '', text)

    text = text.translate(str.maketrans({
        '，': ',', '。': '.', '；': ';', '：': ':', '？': '?', '！': '!',
        '（': '(', '）': ')', '【': '[', '】': ']', '“': '"', '”': '"', '‘': "'", '’': "'"
    }))

    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r"[，。！？；：“”‘’（）【】《》、]", "", text)

    return remove_emoji(text)

def wer_cal(ref, hyp):
    ref = normalize_text(ref.lower(), remove_punctuation=True)
    hyp = normalize_text(hyp.lower(), remove_punctuation=True)
    score = wer(ref, hyp)
    return score

def cer_cal(ref, hyp):
    ref = remove_punctuation_and_whitespace(ref)
    hyp = remove_punctuation_and_whitespace(hyp)
    ref_chars = list(ref.strip().replace(" ", ""))
    hyp_chars = list(hyp.strip().replace(" ", ""))
    r_len = len(ref_chars)

    dp = [[0] * (len(hyp_chars) + 1) for _ in range(r_len + 1)]

    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(len(hyp_chars) + 1):
        dp[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                substitute = dp[i - 1][j - 1] + 1
                insert = dp[i][j - 1] + 1
                delete = dp[i - 1][j] + 1
                dp[i][j] = min(substitute, insert, delete)

    return dp[r_len][len(hyp_chars)] / r_len

def wer_metric(ref, inp_array, asr_model, asr_kwargs, lang):
    inp_wav = torch.tensor(inp_array)
    est_info = asr_model.inference(
        data_in=inp_wav,
        language="auto", # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=False,
        ban_emo_unk=True,
        output_timestamp=False,
        **asr_kwargs,
    )[0][0]
    est_txt = rich_transcription_postprocess(est_info["text"])
    ref = retain_chinese_english(
        remove_punctuation_and_whitespace(
            normalize_text(ref)
        ).upper()
    )
    est_txt = retain_chinese_english(
        remove_punctuation_and_whitespace(
            normalize_text(est_txt)
        ).upper()
    )
    try:
        if lang == "zh":
            ref = zh2ipa(ref)
            est_txt = zh2ipa(est_txt)
            cer_score = cer_cal(ref, est_txt)
        else:
            ref = en2ipa(ref)
            est_txt = en2ipa(est_txt)
            cer_score = wer_cal(ref, est_txt)
        return cer_score, est_txt
    except:
        return 1.0, ""
#########################################################################
def emo_sim(ref, hyp, emo2vec):
    ref = ref.flatten()
    hyp = hyp.flatten()
    e2v_hyp_embs = []
    for hyp_item in slice_audio(hyp):
        hyp_item = torch.tensor(hyp_item, dtype=torch.float32, device="cuda")
        generated_emb = emo2vec.generate(hyp_item, granularity="utterance", extract_embedding=True, disable_pbar=True)[0]["feats"] # 1024
        # RuntimeError: Input type (torch.cuda.DoubleTensor) and weight type (torch.cuda.FloatTensor) should be the same
        e2v_hyp_embs.append(generated_emb)
    e2v_ref_embs = []
    for ref_item in slice_audio(ref):
        ref_item = torch.tensor(ref_item, dtype=torch.float32, device="cuda")
        tgt_emb = emo2vec.generate(ref_item, granularity="utterance", extract_embedding=True, disable_pbar=True)[0]["feats"] # 1024
        e2v_ref_embs.append(tgt_emb)
    generated_emb = np.mean(e2v_hyp_embs, axis=0)
    tgt_emb = np.mean(e2v_ref_embs, axis=0)
    simi = float(F.cosine_similarity(torch.FloatTensor([generated_emb]), torch.FloatTensor([tgt_emb])).item())
    return simi

################################################################
def slice_audio(audio, dim=0):
    if len(audio) > 8*16000:
        subarrays = np.array_split(audio, len(audio) // (8*16000) + 1, axis=dim)
        return subarrays
    else:
        return [audio]

def cos_simi_resemb(model, generated_wav, tgt_wav, sr=16000):
    generated_wav = generated_wav.flatten()
    tgt_wav = tgt_wav.flatten()
    with torch.no_grad():
        gen_wav = preprocess_wav(generated_wav, sr)
        tgt_wav = preprocess_wav(tgt_wav, sr)
        gens = []
        for gen_item in slice_audio(gen_wav):
            gen_embed = model.embed_utterance(gen_item)
            gens.append(gen_embed)
        tgts = []
        for tgt_item in slice_audio(tgt_wav):
            tgt_embed = model.embed_utterance(tgt_item)
            tgts.append(tgt_embed)
        gen_embed = np.mean(gens, axis=0)
        tgt_embed = np.mean(tgts, axis=0)
        sim = F.cosine_similarity(torch.tensor([gen_embed]), torch.tensor([tgt_embed])).item()
        return float(sim)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("filter.filter_api:app", host="0.0.0.0", port=8100, reload=False)