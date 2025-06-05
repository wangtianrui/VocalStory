import os
import qianfan
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import json
class Ernie:
    def __init__(self) -> None:
        os.environ["QIANFAN_AK"] = "f6dzMZACWNJkfiSAVfgCEYBt"
        os.environ["QIANFAN_SK"] = "BKnSCrgBd7gyo0AMKLbwNAbtDiq5hBoL"

    def inference(self, system_prompt, user_prompt, temperature=0.2, model_name="ernie-4.0-turbo-128k"):
        resp = qianfan.ChatCompletion().do(
            endpoint=model_name, 
            messages=[
                {"role":"assistant", "content": system_prompt},
                {"role":"user", "content": user_prompt},
            ], 
            enable_system_memory=False, disable_search=True, enable_citation=False, enable_trace=False,
            temperature=temperature
        )
        return resp["result"]

    def inference_withnetwork(self, system_prompt, user_prompt, temperature=0.2, model_name="ernie-4.0-turbo-128k"):
        resp = qianfan.ChatCompletion().do(
            endpoint=model_name, 
            messages=[
                {"role":"assistant", "content": system_prompt},
                {"role":"user", "content": user_prompt},
            ], 
            enable_system_memory=False, disable_search=False, enable_citation=False, enable_trace=False,
            temperature=temperature
        )
        return resp["result"]

    def inference_withnetwork_multi_round(
        self, system_prompt=None, user_prompt=None, history_inp=None,
        temperature=0.2, model_name="ernie-4.0-turbo-128k"
    ):
        temp_inp = []
        if history_inp is not None:
            for item in history_inp:
                temp_inp.append(item)
        if system_prompt is not None:
            temp_inp.append({"role":"assistant", "content": system_prompt})
        if user_prompt is not None:
            temp_inp.append({"role":"user", "content": user_prompt})
        
        print(temp_inp)
        
        # resp = qianfan.ChatCompletion().do(
        #     endpoint=model_name, 
        #     messages=temp_inp, 
        #     enable_system_memory=False, disable_search=False, enable_citation=False, enable_trace=False,
        #     temperature=temperature
        # )
        url = "https://qianfan.baidubce.com/v2/chat/completions"
        payload = json.dumps({
            "model": model_name,
            "messages": temp_inp,
            "web_search": {
                "enable": True,
                "enable_citation": False,
                "enable_trace": False
            }
        })
        headers = {
            'Content-Type': 'application/json',
            'appid': '',
            'Authorization': 'Bearer bce-v3/ALTAK-sbjCdT2ipwCMH5OJvsrgJ/df060257766fe8ee82e02121c9e00d49fbd2f487'
        }
        
        response = json.loads(requests.request("POST", url, headers=headers, data=payload).text)["choices"][0]["message"]["content"]
        temp_inp.append({"role":"assistant", "content": response})
        return response, temp_inp

app = FastAPI()

class ArrayInput(BaseModel):
    system_prompt: str  
    user_prompt: str 
    temperature: float = 0.2  # 添加类型注解
    model_name: str = "ernie-4.0-turbo-128k"  # 添加类型注解
    
@app.post("/extract_speaker")
def extract_speaker(data: ArrayInput):
    return {
        "result": llm_model.inference(
            system_prompt = data.system_prompt,
            user_prompt = data.user_prompt,
            temperature = data.temperature,
            model_name = data.model_name,
        )
    }

class MultiRoundInput(BaseModel):
    system_prompt: str | None = None
    user_prompt: str | None = None
    history_inp: list[dict] | None = None  # [{"role": ..., "content": ...}, ...]
    temperature: float = 0.2
    model_name: str = "ernie-4.0-turbo-128k"
    
@app.post("/multi_round_chat")
def multi_round_chat(data: MultiRoundInput):
    result, updated_history = llm_model.inference_withnetwork_multi_round(
        system_prompt = data.system_prompt,
        user_prompt = data.user_prompt,
        history_inp = data.history_inp,
        temperature = data.temperature,
        model_name = data.model_name
    )
    return {
        "result": result,
        "updated_history": updated_history
    }
    
llm_model = Ernie()

"""
ernie-4.0-turbo-128k
ernie-4.0-8k
bce-v3/ALTAK-sbjCdT2ipwCMH5OJvsrgJ/df060257766fe8ee82e02121c9e00d49fbd2f487
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("llm.baidu_api:app", host="0.0.0.0", port=8102, reload=False)