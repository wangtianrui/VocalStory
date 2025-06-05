import requests
import torchaudio
import torch
import json

# 构造请求数据
payload = {
    "text": ["笔记它只是一个工具", "真正的目的", "是吸收这些知识"],
    "emotion": ["sad", "angry", "happy"],
    "duration": [3, 2, 2],
    "languages": ["zh", "zh", "zh"],
    "global_emotion": "neutral",
    "speaker": "0004",
    "ssim_threshold": 0.5,
    "esim_threshold": 0.5,
    "wer_threshold": 0.05,
}

# 发送 POST 请求
response = requests.post("http://127.0.0.1:8101/tts", json=payload, proxies={"http": None, "https": None})
print("Status Code:", response.status_code)
# print("Raw Text:", response.text)
print("Headers:", response.headers)
result = json.loads(response.text)
torchaudio.save(r"/Work21/2023/wangtianrui/codes/VocalStory/temp_audio/test.wav", torch.tensor(result["speech"]).unsqueeze(0), result["sample_rate"])
torchaudio.save(r"/Work21/2023/wangtianrui/codes/VocalStory/temp_audio/test_emo.wav", torch.tensor(result["generated_emo_speech"]).unsqueeze(0), result["sample_rate"])