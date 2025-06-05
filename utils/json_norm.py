import json
import ast
import re

def extract_and_convert_json(text):
    def sanitize_text(t):
        """全局替换特殊引号和格式问题，提前做！"""
        t = t.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
        return t

    def extract_json_code_blocks(t):
        """优先提取 markdown 中的 json 代码块"""
        return re.findall(r'```json(.*?)```', t, re.DOTALL)

    def extract_all_json_arrays(t):
        """兜底方案：不管有没有 markdown，匹配所有 JSON 数组"""
        return [match.strip() for match in re.findall(r'\[\s*\{.*?\}\s*\]', t, re.DOTALL)]

    def clean_code_block(t):
        """去掉代码块内多余的空行与首尾空白"""
        return t.strip()

    def remove_comments(t):
        """如果有注释，清除"""
        t = re.sub(r'//.*', '', t)
        t = re.sub(r'/\*.*?\*/', '', t, flags=re.DOTALL)
        t = re.sub(r'#.*', '', t)
        return t

    def try_parse_json(t):
        try:
            return json.loads(t)
        except Exception as e:
            print(f"[JSON Parse Error] {e}")
            return None

    def try_eval(t):
        try:
            obj = ast.literal_eval(t)
            return json.loads(json.dumps(obj, ensure_ascii=False))
        except Exception as e:
            print(f"[Eval Fallback Error] {e}")
            return None

    def process_block(block_text):
        block_text = clean_code_block(block_text)
        block_text = remove_comments(block_text)
        block_text = sanitize_text(block_text)  # ✅ 保险
        result = try_parse_json(block_text)
        if result:
            return result
        result = try_eval(block_text)
        if result:
            return result
        print("❌ 无法解析该代码块：")
        print(block_text[:200])  # 只打印前 200 个字符，防止太长
        return None

    # === 主流程 ===

    # ✅ 1. 预处理：全局 sanitize 一下，避免花引号等问题
    text = sanitize_text(text)

    results = []

    # ✅ 2. 优先尝试 markdown 代码块
    code_blocks = extract_json_code_blocks(text)
    for block in code_blocks:
        parsed = process_block(block)
        if parsed:
            if isinstance(parsed, list):
                results.extend(parsed)
            else:
                results.append(parsed)

    # ✅ 3. 如果 markdown 没找到，尝试所有 JSON 数组
    if not results:
        json_arrays = extract_all_json_arrays(text)
        for block in json_arrays:
            parsed = process_block(block)
            if parsed:
                if isinstance(parsed, list):
                    results.extend(parsed)
                else:
                    results.append(parsed)

    if results:
        print("✅ 最终成功解析 JSON")
        return results
    else:
        print("❌ 最终所有方式解析失败")
        return None