import requests
import numpy as np
import torchaudio

def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav)
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech

# 模拟三个样本，每个样本是一段1秒钟的音频（采样率16kHz）
sample_rate = 16000
duration = 1  # 秒
gen_array = [load_wav(f"/Work21/2023/wangtianrui/codes/fairseq_ecvc/examples/celsds/temp_outputs/zero-shot-ref.wav", 16000).numpy().tolist() for _ in range(3)]
ref_array = [np.random.randn(sample_rate).tolist() for _ in range(3)]
tgt_text = ["笔记他只是一个工具", "真正的目的", "是吸收这些只是"]

# 构造请求数据
payload = {
    "gen_array": gen_array,
    "ref_array": ref_array,
    "tgt_text": tgt_text
}

# 发送 POST 请求
response = requests.post("http://127.0.0.1:8100/predict", json=payload, proxies={"http": None, "https": None})
print("Status Code:", response.status_code)
print("Raw Text:", response.text)
print("Headers:", response.headers)

