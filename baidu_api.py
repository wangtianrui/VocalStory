# 安装包(Python >= 3.7)：pip install qianfan
import os
import qianfan

class Ernie:
    def __init__(self) -> None:
        os.environ["QIANFAN_AK"] = "f6dzMZACWNJkfiSAVfgCEYBt"
        os.environ["QIANFAN_SK"] = "BKnSCrgBd7gyo0AMKLbwNAbtDiq5hBoL"

    def inference(self, system_prompt, user_prompt, temperature=0.2, model_name="ernie-3.5-8k-0613"):
        resp = qianfan.ChatCompletion().do(
            endpoint=model_name, 
            messages=[
                {"role":"system", "content": system_prompt},
                {"role":"user", "content": user_prompt},
            ], 
            enable_system_memory=False, disable_search=False, enable_citation=False, enable_trace=False,
            temperature=temperature
        )
        return resp["result"]

