import os
import numpy as np
from tqdm import tqdm
import ast
from argparse import Namespace
from itertools import chain
from omegaconf import DictConfig
import string
punctuation_string = string.punctuation
import re
import soundfile as sf
import jieba
from pypinyin import pinyin, Style
import math
from g2p_en import G2p
from num2words import num2words

def convert_sil(phones):
    phones = phones.replace("sil_S", "sil")
    phones = phones.replace("sil_B", "sil")
    phones = phones.replace("sil_I", "sil")
    phones = phones.replace("sil_E", "sil")
    return phones

def convert_er(phones):
    phones = phones.replace("ə_B r_I", "ər_B")
    phones = phones.replace("ə_I r_I", "ər_I")
    phones = phones.replace("ə_I r_E", "ər_E")
    phones = phones.replace("ə_B r_E", "ər_S")
    return phones

def arpa2ipa(phones, arpa2ipa_dict):
    if isinstance(phones, list):
        # phones are from g2p
        if len(phones) == 1:
            phones[0] = phones[0] + "_S"
        else:
            head_phone = phones[0]
            tail_phone = phones[-1]
            phones = [p + "_I" for p in phones]
            phones[0] = head_phone + "_B"
            phones[-1] = tail_phone + "_E"
        phones = [p for p in phones if p in arpa2ipa_dict]
    else:
        phones = phones.split()
    phones = [arpa2ipa_dict[p] for p in phones]
    return " ".join(phones)

def load_zh2ipa(args):
    zh_wrd_to_phn = {}
    with open(args.pinyin2pho, "r") as lf:
        items = [line.strip().split('\t') for line in lf]
        piny_to_phn = dict(items)
    polyphones = []
    with open(args.lexicon, "r") as lf:
        for line in tqdm(lf, maxinterval=1):
            items = line.rstrip().split(' ', 2)
            assert len(items) > 1, line
            if items[0] in zh_wrd_to_phn:
                polyphones.append(items)
            zh_wrd_to_phn[items[0]] = items[2]
    return zh_wrd_to_phn, piny_to_phn

def load_en2ipa(args):
    wrd_to_phn1 = {}
    wrd_to_phn2 = {}
    g2p = G2p()
    with open(args.arpa2ipa, "r") as lf:
        arpa2ipa_dict = [l.strip().split() for l in lf]
        arpa2ipa_dict = dict(arpa2ipa_dict)

    polyphones = []
    with open(args.librispeech_lexicon, "r") as lf:
        for line in tqdm(lf, maxinterval=1):
            items = line.rstrip().split(' ', 2)
            assert len(items) > 1, line
            if items[0] in wrd_to_phn1:
                polyphones.append(items)
            wrd_to_phn1[items[0]] = arpa2ipa(items[2], arpa2ipa_dict)
    
    polyphones2 = []
    with open(args.wenetspeech_lexicon, "r") as lf:
        for line in tqdm(lf, maxinterval=1):
            items = line.rstrip().split(' ', 2)
            assert len(items) > 1, line
            if items[0] in wrd_to_phn2:
                polyphones2.append(items)
            wrd_to_phn2[items[0]] = convert_er(items[2])
    return wrd_to_phn1, wrd_to_phn2, g2p, arpa2ipa_dict

def en2ipa(line):
    line = line.strip().upper()
    line = re.sub(u"([^\u0041-\u005a\u0061-\u007a\u0030-\u0039\'])", " ", line)
    items = []
    for item in line.split():
        if item.isdigit():
            try:
                item = num2words(item)
            except Exception as e:
                print(line, str(e))
        items.append(item)
    line = " ".join(items)
    line = line.replace("-", " ")
    line = line.upper()
    line = line.replace("' S", "'S")
    words = line.strip().upper().split()
    phones = []
    for i, w in enumerate(words):
        # first, try librispeech_lexicon
        if w in wrd_to_phn1:
            phones.append(wrd_to_phn1[w])
        # then, try wenetspeech_lexicon
        elif w in wrd_to_phn2:
            phones.append(wrd_to_phn2[w])
        # at last, use g2p
        else:
            try:
                phone = g2p(w)
                if len(phone) == 0:
                    print(f"| Warning: {w} g2p results: {phone}")
                    phone = ''
                else:
                    phone = arpa2ipa(phone, arpa2ipa_dict)
                phones.append(phone)
                wrd_to_phn1[w] = phone
            except Exception as e:
                print(e)
                import pdb
                pdb.set_trace()
        
    phone_line = convert_sil(" ".join(phones))
    return phone_line


def pinyin2ipa(piny, piny_to_phn):
    phones = []
    for i, w in enumerate(piny):
        if w[-1].isdigit():
            tone = w[-1]
            w = w[:-1]
            if tone == '5':
                tone = '0'
            if w == 'n':
                w = 'en' # n/en in different pinyin system
            if w in piny_to_phn:
                phone = piny_to_phn[w] + '_' + tone
                phones.extend(phone.split())
            else:
                # e.g. 'eng' not in standard pinyin2phone
                phones.append("<unk>")
        else:
            phones.append("<unk>")
        
    num = len(phones)
    for i in range(num):
        if not phones[i] == "<unk>":
            if i == 0:
                phones[i] += "_B"
            elif i == num - 1:
                phones[i] += "_E"
            else:
                phones[i] += "_I"
    
    return " ".join(phones)

def process_zh_line(line):
    words = line.strip().upper().split()
    phones = []
    for i, w in enumerate(words):
        # first, try wenetspeech_lexicon
        if w in zh_wrd_to_phn:
            phone = zh_wrd_to_phn[w]
        # then, try pinyin
        else:
            piny = pinyin(w, style=Style.TONE3, neutral_tone_with_five=True)
            piny = [i[0] for i in piny if i[0] !=' ']
            if len(piny) == 0:
                print(f"| Warning: pinyin of {w} is {piny}")
                phone = "<unk>"
            else:
                phone = pinyin2ipa(piny, piny_to_phn)
            zh_wrd_to_phn[w] = phone
        if phone.find("<unk>") >= 0:
            pass
        phones.append(phone)
    phone_line = convert_sil(" ".join(phones))
    return phone_line

def zh2ipa(line):
    line = line.strip()
    line = line.upper()
    line = line.replace("-", "")
    norm_line = re.sub(u"([^0-9a-zA-Z\u4e00-\u9fa5])", " ", line)
    tokenized_words = [w for w in jieba.cut(norm_line) if w !=' ']
    
    ipa_line = process_zh_line(
        " ".join(tokenized_words)
    )
    return ipa_line

# english
wrd_to_phn1 = {}
wrd_to_phn2 = {}
g2p = G2p()
with open(r"model_cache/ipa/ARPA2IPA.map", "r") as lf:
    arpa2ipa_dict = [l.strip().split() for l in lf]
    arpa2ipa_dict = dict(arpa2ipa_dict)

polyphones = []
with open(r"model_cache/ipa/librispeech_lexicon.txt", "r") as lf:
    for line in tqdm(lf, maxinterval=1):
        items = line.rstrip().split(' ', 2)
        assert len(items) > 1, line
        if items[0] in wrd_to_phn1:
            polyphones.append(items)
        wrd_to_phn1[items[0]] = arpa2ipa(items[2], arpa2ipa_dict)

polyphones2 = []
with open(r"model_cache/ipa/wenetspeech_align_lexicon.txt", "r") as lf:
    for line in tqdm(lf, maxinterval=1):
        items = line.rstrip().split(' ', 2)
        assert len(items) > 1, line
        if items[0] in wrd_to_phn2:
            polyphones2.append(items)
        wrd_to_phn2[items[0]] = convert_er(items[2])


# Chinese
zh_wrd_to_phn = {}
with open(r"model_cache/ipa/pinyin2ipa.txt", "r") as lf:
    items = [line.strip().split('\t') for line in lf]
    piny_to_phn = dict(items)
polyphones = []
with open(r"model_cache/ipa/wenetspeech_align_lexicon.txt", "r") as lf:
    for line in tqdm(lf, maxinterval=1):
        items = line.rstrip().split(' ', 2)
        assert len(items) > 1, line
        if items[0] in zh_wrd_to_phn:
            polyphones.append(items)
        zh_wrd_to_phn[items[0]] = items[2]

# if __name__ == "__main__":
#     en_test = r"hello world"
#     zh_test = r"你好世界！"
#     print(process_en(en_test))
#     print(process_zh(zh_test))