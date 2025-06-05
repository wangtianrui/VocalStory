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

css = """
.step-heading {font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem}
"""

# 保存路径配置
formatted_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
BOOK_TEMP_PATH = f"./outputs/{formatted_time}/"
CHARACTER_JSON_PATH = f"./outputs/{formatted_time}/extracted_characters.json"
SCRIPT_JSON_PATH = f"./outputs/{formatted_time}/script.json"
LLM_HISTORY_JSON_PATH = f"./outputs/{formatted_time}/llm_history.json"
os.makedirs(BOOK_TEMP_PATH)

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
# def tts():
#     with open(SCRIPT_JSON_PATH, "r") as rf:
#         script_json = json.load(rf)
#     with open(CHARACTER_JSON_PATH, "r") as rf:
#         character_json = json.load(rf)
    
#     for item

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
            gr.Markdown('<div class="step-heading">🎨 Step 4: Script Extraction and Refining</div>')
    #         with gr.Accordion("Editing Tips", open=True):
    #             gr.Markdown("""
    #             * Remove unwanted sections: Table of Contents, About the Author, Acknowledgements
    #             * Fix formatting issues or OCR errors
    #             * Check for chapter breaks and paragraph formatting
    #             """)
            
    #         text_output = gr.Textbox(
    #             label="Edit Book Content", 
    #             placeholder="Extracted text will appear here for editing",
    #             interactive=True, 
    #             lines=15
    #         )
            
    #         save_btn = gr.Button("Save Edited Text", variant="primary")

    # with gr.Row():
    #     with gr.Column():
    #         gr.Markdown('<div class="step-heading">🧩 Step 3: Character Identification (Optional)</div>')
            
    #         identify_btn = gr.Button("Identify Characters", variant="primary")
            
    #         with gr.Accordion("Why Identify Characters?", open=True):
    #             gr.Markdown("""
    #             * Improves multi-voice narration by assigning different voices to characters
    #             * Creates more engaging audiobooks with distinct character voices
    #             * Skip this step if you prefer single-voice narration
    #             """)
                
    #         character_output = gr.Textbox(
    #             label="Character Identification Progress", 
    #             placeholder="Character identification progress will be shown here",
    #             interactive=False,
    #             lines=3
    #         )

    # with gr.Row():
    #     with gr.Column():
    #         gr.Markdown('<div class="step-heading">🎧 Step 4: Generate Audiobook</div>')
            
    #         with gr.Row():
    #             voice_type = gr.Radio(
    #                 ["Single Voice", "Multi-Voice"], 
    #                 label="Narration Type",
    #                 value="Single Voice",
    #                 info="Multi-Voice requires character identification"
    #             )

    #             narrator_gender = gr.Radio(
    #                 ["male", "female"], 
    #                 label="Choose whether you want the book to be read in a male or female voice",
    #                 value="female"
    #             )
                
    #             output_format = gr.Dropdown(
    #                 ["M4B (Chapters & Cover)", "AAC", "M4A", "MP3", "WAV", "OPUS", "FLAC", "PCM"], 
    #                 label="Output Format",
    #                 value="M4B (Chapters & Cover)",
    #                 info="M4B supports chapters and cover art"
    #             )
            
    #         generate_btn = gr.Button("Generate Audiobook", variant="primary")
            
    #         audio_output = gr.Textbox(
    #             label="Generation Progress", 
    #             placeholder="Generation progress will be shown here",
    #             interactive=False,
    #             lines=3
    #         )
            
    #         # Add a new File component for downloading the audiobook
    #         with gr.Group(visible=False) as download_box:
    #             gr.Markdown("### 📥 Download Your Audiobook")
    #             audiobook_file = gr.File(
    #                 label="Download Generated Audiobook",
    #                 interactive=False,
    #                 type="filepath"
    #             )
    
    # Connections with proper handling of Gradio notifications
    # validate_btn.click(
    #     validate_book_upload, 
    #     inputs=[book_input, book_title], 
    #     outputs=[]
    # )
    
    # convert_btn.click(
    #     text_extraction_wrapper, 
    #     inputs=[book_input, text_decoding_option, book_title], 
    #     outputs=[text_output],
    #     queue=True
    # )
    
    # save_btn.click(
    #     save_book_wrapper, 
    #     inputs=[text_output, book_title], 
    #     outputs=[],
    #     queue=True
    # )
    
    # identify_btn.click(
    #     identify_characters_wrapper, 
    #     inputs=[book_title], 
    #     outputs=[character_output],
    #     queue=True
    # )
    
    # # Update the generate_audiobook_wrapper to output both progress text and file path
    # generate_btn.click(
    #     generate_audiobook_wrapper, 
    #     inputs=[voice_type, narrator_gender, output_format, book_input, book_title], 
    #     outputs=[audio_output, audiobook_file],
    #     queue=True
    # ).then(
    #     # Make the download box visible after generation completes successfully
    #     lambda x: gr.update(visible=True) if x is not None else gr.update(visible=False),
    #     inputs=[audiobook_file],
    #     outputs=[download_box]
    # )
    
app = FastAPI()
app = gr.mount_gradio_app(app, gradio_app, path="/")  # Mount Gradio at root

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)