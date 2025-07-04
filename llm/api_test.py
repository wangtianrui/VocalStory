import requests
import torchaudio
import torch
import json

# 构造请求数据
# payload = {
#     "system_prompt": """
# 你是一名广播剧编辑，你可以基于输入的小说标题，小说的章节内容，规划出一个多角色的广播剧剧本。以下是你操作的具体步骤：
# 1. 你会根据小说的标题去搜索其中已知的关键角色，以及角色信息；
# 2. 你会将我输入的章节内容中所涉及的角色全部地提取出来，并确定以及推测他们的信息；
# 3. 年龄和性别是必须的有的，并且年龄的取值范围在10~80之间的具体的一个数值，性别和年龄你可以通过名字和上下文进行适当的推断；
# 4. 如果是非人类的生物，也需要你根据你的知识结合故事（如玄幻题材等）给出这个生物的性别和年龄，甚至可以是随机猜一个，不能写未知或者unknown之类的；
# 5. 只需要给我json的结果，不要有注释，也不要有任何的解释。

# 最终的结果格式是一个json，key是人名，如：
# {
#     "Emili": { "age": 19, "gender": "female" },
#     "Bob": { "age": 34, "gender": "male" },
#     "一群人": { "age": "5~30", "gender": "mix" },
#     "小鸟": { "age": "8", "gender": "female" },
# }
# """,
#     "user_prompt": """
# 我输入的小说标题是：《完美世界》
# 我输入的小说章节内容是：
# 第一章 朝气蓬勃
# 石村，位于苍莽山脉中，四周高峰大壑，茫茫群山巍峨。
# 清晨，朝霞灿灿，仿若碎金一般洒落，沐浴在人身上暖洋洋。
# 一群孩子，从四五岁到十几岁不等，能有数十人，在村前的空地上迎着朝霞，正在哼哈有声的锻炼体魄。一张张稚嫩的小脸满是认真之色，大一些的孩子虎虎生风，小一些的也比划的有模有样。
# 一个肌体强健如虎豹的中年男子，穿着兽皮衣，皮肤呈古铜色，黑发披散，炯炯有神的眼眸扫过每一个孩子，正在认真指点他们。
# “太阳初升，万物初始，生之气最盛，虽不能如传说中那般餐霞食气，但这样迎霞锻体自也有莫大好处，可充盈人体生机。一天之计在于晨，每日早起多用功，强筋壮骨，活血炼筋，将来才能在这苍莽山脉中有活命的本钱。”站在前方、指点一群孩子的中年男子一脸严肃，认真告诫，而后又喝道：“你们明白吗？”
# “明白！”一群孩子中气十足，大声回应。
# 山中多史前生物出没，时有遮蔽天空之巨翼横过，在地上投下大片的阴影，亦有荒兽立于峰上，吞月而啸，更少不了各种毒虫伏行，异常可怖。
# “明白呀。”一个明显走神、慢了半拍的小家伙奶声奶气的叫道。
# 这是一个很小的孩子，只有一两岁的样子，刚学会走路没几个月，也在跟着锻炼体魄。显然，他是自己凑过来的，混在了年长的孩子中，分明还不应该出现在这个队伍里。
# “哼哼哈嘿！”小家伙口中发声，嫩嫩的小手臂卖力的挥动着，效仿大孩子们的动作，可是他太过幼小，动作歪歪扭扭，且步履蹒跚，摇摇摆摆，再加上嘴角间残留的白色奶渍，引人发笑。
# 一群大孩子看着他，皆挤眉弄眼，让原本严肃的晨练气氛轻缓了不少。
# 小不点长的很白嫩与漂亮，大眼睛乌溜溜的转动，整个人像是个白瓷娃娃，很可爱，稚嫩的动作，口中咿咿呀呀，憨态可掬。这让另一片场地中盘坐在一块块巨石上正在吞吐天精的一些老人也都露出笑容。
# 就是那些身材高大魁梧、上半身赤裸、肌腱光亮并隆起的成年男子们，也都望了过来，带着笑意。他们是村中最强壮的人，是狩猎与守护这个村落的最重要力量，也都在锻体，有人握着不知名的巨兽骨骼打磨而成的白骨大棒，也有人持着黑色金属铸成的阔剑，用力舞动，风声如雷。
# 生存环境极其恶劣，多洪荒猛兽毒虫，为了食物，为了生存，很多男子还未成年就过早夭折在了大荒中，想要活下去，唯有强壮己身。清晨用功，无论是成年人，亦或是老人与孩子，这是每一个人自幼就已养成的习惯。
# “收心！”负责督促与指导孩子练功的中年男子大声喊道。一群孩子赶紧认真了起来，继续在柔和与灿烂的朝霞中锻炼。
# “呼……咿呀，累了。”小不点长出了一口气，一屁墩儿坐在了地上，看着大孩子们锻炼体魄。可仅一会儿工夫他就被分散了注意力，站起身来，摇摇摆摆，冲向不远处一只正在蹦蹦跳跳的五色雀，结果磕磕绊绊，连摔了几个屁墩儿，倒也不哭，气呼呼，哼哼唧唧爬起来再追。
# """,
#     "temperature": 0.2,
# }

# # 发送 POST 请求
# response = requests.post("http://127.0.0.1:8102/extract_speaker", json=payload, proxies={"http": None, "https": None})
# print("Status Code:", response.status_code)
# result = json.loads(response.text)["result"]
# print("Raw Text:", result)
# print("Headers:", response.headers)

def run_multi_turn_chat(prompt, history):
    if not history:
        history = [{
            "role":"assistant", 
            "content": "1+1等于几？"
        }]
    payload = {
        "system_prompt": None,
        "user_prompt": prompt,
        "history_inp": history,
        "temperature": 0.2,
        "model_name": "ernie-4.0-8k"
    } 
    response = requests.post("http://127.0.0.1:8102/multi_round_chat", json=payload, proxies={"http": None, "https": None})
    response = json.loads(response.text)
    print(response)
    result = response["result"]
    updated_history = response["updated_history"]
    return result, updated_history

history = []
re, history = run_multi_turn_chat("你好啊", history)
print(re)
re, history = run_multi_turn_chat("刚刚我问你的算法是啥？", history)
print(re)