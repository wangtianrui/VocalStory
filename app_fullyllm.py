import gradio as gr
import os
import traceback
from fastapi import FastAPI
import requests
import json
import utils.preset_infos as preset_infos
from datetime import datetime
import re
import random
import torchaudio
import torch

css = """
.step-heading {font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem}
"""

# 保存路径配置
formatted_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
BOOK_TEMP_PATH = f"./outputs/{formatted_time}/"
CHARACTER_JSON_PATH = f"./outputs/{formatted_time}/extracted_characters.json"
SCRIPT_JSON_PATH = f"./outputs/{formatted_time}/script.json"
LLM_HISTORY_JSON_PATH = f"./outputs/{formatted_time}/llm_history.json"
WAV_SAVE_HOME = f"./outputs/{formatted_time}/wavs/"
os.makedirs(BOOK_TEMP_PATH, exist_ok=True)
os.makedirs(WAV_SAVE_HOME, exist_ok=True)

################内容提取##################

def clean_json_result(response_text: str):
    # 去掉 markdown 包裹
    cleaned = re.sub(r"^```json\n", "", response_text)
    cleaned = re.sub(r"\n```$", "", cleaned)
    return cleaned  # 不解码，不处理 unicode_escape

# 处理图书上传并提取前2000字符
def extract_book_content(book_file):
    if book_file is None:
        return gr.update(value="", visible=True), "⚠️ 请先上传图书文件。"

    with open(book_file.name, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    truncated = content[:2000]
    msg = ""
    if len(content) > 2000:
        msg = "⚠️ 图书内容超过2000字符，仅显示前2000字符。"

    return gr.update(value=truncated), msg

################角色提取##################

# 保存编辑后的图书文本并提取角色
def save_book_and_extract_characters(
    book_title_text, edited_text, narra_age, narra_gender
):
    global BOOK_TEMP_PATH
    if not BOOK_TEMP_PATH.endswith(".txt"):
        BOOK_TEMP_PATH = os.path.join(BOOK_TEMP_PATH, f"{book_title_text.strip()}.txt")
    with open(BOOK_TEMP_PATH, "w", encoding="utf-8") as f:
        f.write(edited_text)

    try:
        raw_result = extract_character(book_title_text, edited_text)
        cleaned_result = clean_json_result(raw_result)
        # 解析成 dict 检查合法性
        characters = json.loads(cleaned_result)
        # 重新格式化为正常带换行的 JSON 字符串
        characters["Narrator"] = {"age": narra_age, "gender": narra_gender}
        formatted_result = json.dumps(characters, ensure_ascii=False, indent=2)
        return formatted_result, "✅ 角色提取成功！"
    except Exception as e:
        traceback.print_exc()
        return "", f"❌ 角色提取失败：{e}"

# Character Extraction
def extract_character(title, content, temperature=0.5):
    payload = {
        "system_prompt": preset_infos.CHINESE_CHARACTER_SYSTEMPROMPT,
        "user_prompt": preset_infos.CHINESE_CHARACTER_USERPROMPT%(title, content),
        "temperature": temperature,
    }
    # 发送 POST 请求
    response = requests.post("http://127.0.0.1:8102/extract_speaker", json=payload, proxies={"http": None, "https": None})
    print("Status Code:", response.status_code)
    result = json.loads(response.text)["result"]
    return result # text

# 校验并保存角色JSON
def save_character_json(character_text, lang, top_k):
    try:
        parsed_json = json.loads(character_text)
        with open(CHARACTER_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(parsed_json, f, ensure_ascii=False, indent=2)
        speaker_info = assign_speaker_info_flexible_gender(lang, top_k)
        speaker_info = json.dumps(speaker_info, ensure_ascii=False, indent=2)
        
        with open(BOOK_TEMP_PATH, "r", encoding="utf-8") as f:
            content = "\n".join(f.readlines())
            title = os.path.basename(BOOK_TEMP_PATH).replace(".txt", "")
        llm_1st_inp = preset_infos.CHINESE_SCRIPT_USERPROMPT%(
            title, content, speaker_info
        )
        return speaker_info, llm_1st_inp, "✅ 人物JSON保存成功！"
    except json.JSONDecodeError as e:
        return None, f"❌ JSON格式错误，请检查：{e}"

# 根据TTS的角色库,分配prompt speaker
def assign_speaker_info_flexible_gender(lang='zh', top_k=10):
    with open(CHARACTER_JSON_PATH, "r") as rf:
        speaker_info = json.load(rf)
    response = requests.post("http://127.0.0.1:8101/speaker_list", proxies={"http": None, "https": None})
    candidates = json.loads(response.text)
    
    used_ids = set()

    # 只保留符合 lang 的候选
    lang_matched = {
        sid: info for sid, info in candidates.items()
        if info["lang"] == lang
    }

    for role, role_info in speaker_info.items():
        if "speaker_id" in speaker_info[role].keys():
            used_ids.add(speaker_info[role]["speaker_id"])
            continue
        role_gender = role_info.get("gender")
        role_age = role_info.get("age")

        # 处理年龄（支持字符串）
        if isinstance(role_age, str):
            if "~" in role_age:
                try:
                    avg_age = sum(map(float, role_age.split("~"))) / 2
                except:
                    avg_age = 30
            else:
                try:
                    avg_age = float(role_age)
                except:
                    avg_age = 30
        else:
            avg_age = float(role_age)

        # 如果性别是 mix 或非法，则从 lang_matched 中随机选一个性别
        valid_genders = {"male", "female"}
        if role_gender not in valid_genders:
            # 获取当前 lang 下可用的性别集合
            gender_pool = set([v["gender"] for v in lang_matched.values()])
            selectable_gender = random.choice(list(gender_pool & valid_genders))
            role_gender = selectable_gender

        # 筛选 lang + gender 匹配的 speaker
        filtered = {
            sid: info for sid, info in lang_matched.items()
            if info["gender"] == role_gender
        }

        if not filtered:
            raise ValueError(f"❌ 无法找到语言为 {lang} 且性别为 {role_gender} 的 speaker，用于角色：{role}")

        # 按年龄差排序
        sorted_candidates = sorted(
            filtered.items(),
            key=lambda item: abs(item[1]["age"] - avg_age)
        )

        # 优先选未用过的最接近项
        for sid, info in sorted_candidates:
            if sid not in used_ids:
                speaker_info[role]["speaker_id"] = sid
                speaker_info[role]["speaker_ref"] = info
                used_ids.add(sid)
                break
        else:
            # 如果全都用过，从前 top_k 中选一个未用过的
            top_k_unused = [
                (sid, info) for sid, info in sorted_candidates[:top_k]
                if sid not in used_ids
            ]
            if top_k_unused:
                sid, info = random.choice(top_k_unused)
                speaker_info[role]["speaker_id"] = sid
                speaker_info[role]["speaker_ref"] = info
                used_ids.add(sid)
            else:
                # 最终允许从前 top_k 中随机复用（允许重复）
                sid, info = random.choice(sorted_candidates[:top_k])
                speaker_info[role]["speaker_id"] = sid
                speaker_info[role]["speaker_ref"] = info
    with open(CHARACTER_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(speaker_info, f, ensure_ascii=False, indent=2)
    return speaker_info

#################剧本生成#################
# LLM生成剧本
def run_multi_turn_chat(prompt, history):
    if not history:
        history = [{
            "role":"assistant", 
            "content": preset_infos.CHINESE_SCRIPT_SYSTEMPROMPT
        }]
    payload = {
        "user_prompt": prompt,
        "history_inp": history,
        "temperature": 0.8,
        "model_name": "ernie-4.0-8k"
    }
    response = requests.post("http://127.0.0.1:8102/multi_round_chat", json=payload, proxies={"http": None, "https": None})
    response = json.loads(response.text)
    print(response)
    result = response["result"]
    updated_history = response["updated_history"]
    return result, updated_history

# 保存剧本json
def save_script_json(script_output_text, history):
    try:
        parsed_script = json.loads(script_output_text)

        with open(SCRIPT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(parsed_script, f, ensure_ascii=False, indent=2)
        
        with open(LLM_HISTORY_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        return f"✅ Script saved"
    except json.JSONDecodeError as e:
        return f"❌ JSON format error in script: {e}"

#################TTS#################
def generation():
    with open(SCRIPT_JSON_PATH, "r") as rf:
        script_json = json.load(rf)
    with open(CHARACTER_JSON_PATH, "r") as rf:
        character_json = json.load(rf)
    
    for i, item in enumerate(script_json):
        save_name = os.path.join(WAV_SAVE_HOME, f"{i}.wav")
        if os.path.exists(save_name):
            continue
        payload = {
            "text": [tmp["line"] for tmp in item["lines"]],
            # "emotion": [emo_dict[tmp["emotion"]] for tmp in item["lines"]],
            "emotion": [tmp["emotion"] for tmp in item["lines"]],
            "duration": [float(tmp["duration"]) for tmp in item["lines"]],
            "languages": ["zh" for tmp in item["lines"]],
            "global_emotion": item["global_emotion"],
            "speaker": character_json[item["speaker"]]["speaker_id"],
            "ssim_threshold": 0.5,
            "esim_threshold": 0.5,
            "wer_threshold": 0.05,
        }
        if item["speaker"] == "Narrator":
            response = requests.post("http://127.0.0.1:8103/tts_narration", json=payload, proxies={"http": None, "https": None})
        else:
            response = requests.post("http://127.0.0.1:8101/tts_genshin", json=payload, proxies={"http": None, "https": None})
        result = json.loads(response.text)
        torchaudio.save(save_name, torch.tensor(result["speech"]).unsqueeze(0), result["sample_rate"])

    # ✅ 拼接所有 wav 并保存为 all.wav
    combined_audio = []

    for i in range(len(script_json)):
        wav_path = os.path.join(WAV_SAVE_HOME, f"{i}.wav")
        if os.path.exists(wav_path):
            waveform, sr = torchaudio.load(wav_path)
            combined_audio.append(waveform)
            combined_audio.append(torch.zeros(1, int(sr*0.5)))

    # 拼接所有段落
    if combined_audio:
        final_audio = torch.cat(combined_audio, dim=1)  # 在时间维度拼接
        final_save_path = os.path.join(WAV_SAVE_HOME, "all.wav")
        torchaudio.save(final_save_path, final_audio, sr)
        print(f"✅ All audio saved to: {final_save_path}")
    else:
        final_save_path = None
        print("❌ No audio segments to merge.")
    return final_save_path
##################################
with gr.Blocks(css=css, theme=gr.themes.Default()) as gradio_app:
    gr.Markdown("# 📖 Audiobook Creator")
    gr.Markdown("Create professional audiobooks from your ebooks in just a few steps.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('<div class="step-heading">📚 Step 1: Book Details</div>')

            book_title = gr.Textbox(
                label="Book Title",
                placeholder="Enter the title of your book"
            )

            book_input = gr.File(
                label="Upload Book",
                height=150,
                file_types=[".txt"]
            )

            extract_book_btn = gr.Button("Extract Book", variant="primary")
            book_extraction_notice = gr.Markdown("", visible=False)

            with gr.Accordion("Editing Tips", open=True):
                gr.Markdown("""
                * Remove unwanted sections: Table of Contents, About the Author, Acknowledgements  
                * Fix formatting issues or content errors  
                * Check for chapter breaks and paragraph formatting  
                """)
            text_output = gr.Textbox(
                label="Edit Book Content",
                placeholder="Extracted text will appear here for editing",
                interactive=True,
                lines=20
            )
            with gr.Row():
                narra_age = gr.Number(
                    label="Narrator Age",
                    value=30,
                    precision=0,
                    scale=1
                )
                narra_gender = gr.Dropdown(
                    label="Narrator Gender",
                    choices=["male", "female"],
                    value="male",
                    scale=1
                )
            save_book_extract_char_btn = gr.Button("Save Edited Book Content and Extract Characters", variant="primary")
            
    with gr.Row():
        with gr.Column():
            gr.Markdown('<div class="step-heading">🪪 Step 2: Character Refining</div>')

            character_output = gr.Textbox(
                label="Edit Character Json",
                placeholder="Extracted characters will appear here for editing",
                interactive=True,
                lines=20
            )

            with gr.Row():
                lang_select = gr.Dropdown(
                    label="Language for Speaker Matching",
                    choices=["zh", "en"],
                    value="zh"
                )

                topk_slider = gr.Slider(
                    label="Top-K Candidates for Random Selection",
                    minimum=2,
                    maximum=20,
                    step=1,
                    value=10
                )
            save_character_btn = gr.Button("Match Speaker Database and Save Characters", variant="primary")
            
            character_save_notice = gr.Markdown("", visible=False)
    
    with gr.Row():
        with gr.Column(scale=1):  # 左侧展示区
            
            gr.Markdown('<div class="step-heading">🗓️ Step 3: Script Extraction and Refining</div>')
            script_output = gr.Textbox(
                label="Script Output (LLM Result)",
                interactive=True,
                lines=20
            )
            prompt_input = gr.Textbox(
                label="Editable Prompt",
                placeholder="Enter your prompt here...",
                interactive=True,
                lines=5
            )
            script_save_notice = gr.Markdown("")
            with gr.Row():
                run_llm_btn = gr.Button("🔁 Run LLM to Modify Script")
                save_script_btn = gr.Button("💾 Save Script Output", variant="primary")
    
    llm_history_state = gr.State([]) 
    
    # 提取书内容
    extract_book_btn.click(
        extract_book_content,
        inputs=[book_input],
        outputs=[text_output, book_extraction_notice]
    )
    # 抽取角色
    save_book_extract_char_btn.click(
        save_book_and_extract_characters,
        inputs=[book_title, text_output, narra_age, narra_gender],
        outputs=[character_output, character_save_notice]
    )
    # 保存角色
    save_character_btn.click(
        save_character_json,
        inputs=[character_output, lang_select, topk_slider],
        outputs=[character_output, prompt_input, character_save_notice]
    )
    # LLM对话生成剧本
    run_llm_btn.click(
        run_multi_turn_chat,
        inputs=[prompt_input, llm_history_state],
        outputs=[script_output, llm_history_state]
    )
    # 保存剧本
    save_script_btn.click(
        save_script_json,
        inputs=[script_output, llm_history_state],
        outputs=[script_save_notice]
    )
    
    with gr.Row():
        with gr.Column(scale=1):  # 左侧展示区
            gr.Markdown('<div class="step-heading">🎨 Step 4: AudioBook Generation</div>')
            # 生成按钮
            generate_button = gr.Button("生成")

            # 播放器，初始为空
            audio_player = gr.Audio(label="生成的语音", type="filepath")

            # 绑定按钮点击事件
            generate_button.click(
                fn=generation,          # 调用的函数
                inputs=[],              # 无输入
                outputs=[audio_player]  # 输出给播放器
            )
    
app = FastAPI()
app = gr.mount_gradio_app(app, gradio_app, path="/")  # Mount Gradio at root

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7861)