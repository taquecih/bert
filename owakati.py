#!/usr/bin/env python
# # coding: utf8

import MeCab
import re

n = 4
cat = ["B-OG", "B-UG", "B-MT", "B-GM", "B-LC"]
fn = ["tag11.txt", "tag12.txt", "tag13.txt", "tag23.txt", "tag31.txt"]

#タグ名指定
b = cat[n]
f2n = "sent_" + b + ".txt"
f3n = "tags_" + b + ".txt"

f = open(fn[n], 'r', encoding="utf-8")
l = f.readlines()
o = [0] * len(l)
t = [0] * len(l)

wakati = MeCab.Tagger("-Owakati")

#分かたれたタグを復元
def tag_re(str, s1="\< ?(\/?) ?(\d\d) ?\>", s2="\<(\d\d)\> ", s3=" \<\/(\d\d)\>"):
    str = re.sub(s1, "<\\1\\2>", str)
    str = re.sub(s2, "<\\1>", str)
    str = re.sub(s3, "</\\1>", str)
    str = re.sub("([^ ])\<(\d\d)\>", "\\1 <\\2>", str)
    str = re.sub("\<\/(\d\d)\>([^ ])", "</\\1> \\2", str)
    str = re.sub("(\d) \. (\d)", "\\1.\\2", str)
    str = re.sub("([^ \>])、", "\\1 、", str)
    return str

#タグをrangeからtokenに付け直す
def totag(str):
    str = re.sub("\<\d\d\>", "＠", str)
    str = re.sub("\<\/\d\d\>", "￥", str)
    mch = re.findall("＠.*?￥", str)
    rep = [0] * len(mch)
    for i in range(len(mch)):
        rep[i] = mch[i].replace(" ", "￥ ＠")
        str = str.replace(mch[i], rep[i])
    return str

#owakatiの後処理
def preed(str):
    str = re.sub(" {1,}", " ", str)
    str = re.sub(" *\t *", "\t", str)
    str = re.sub(" $", "", str)
    return str

#最後の処理
def psted(str):
    str = str.replace("＠", " ").replace("￥", " ")
    str = re.sub(" {1,}", " ", str)
    str = re.sub(" *\t *", "\t", str)
    str = re.sub(" $", "", str)
    return str

#tagだけの配列を生成
def sent2tags(str):
    str = re.sub("[^＠￥ \t\n]+", "O", str)
    str = str.replace("＠O￥", b).replace("\t", " ").replace("\n", "")
    return str

#英語の最小限のtokenize
def minitokenizer(str):
    str = re.sub("([\u0021-\u002f:;=?\[/\]^_{}|~—])", " \\1 ", str)
    str = re.sub(" {2,}", " ", str)
    return str

for i in range(len(l)):
    s = l[i].split("\t")
    sent = minitokenizer(s[0]) + "\t" + wakati.parse(s[1])
    sent = tag_re(sent)
    sent = totag(sent)
    sent = preed(sent)
    tags = sent2tags(sent)
    sent = psted(sent)
    o[i] = sent
    t[i] = tags

o = "".join(o)
t = "\n".join(t)

f2 = open(f2n, 'w', encoding="utf-8")
f2.write(o)
f2.close()

f3 = open(f3n, 'w', encoding="utf-8")
f3.write(t)
f3.close()
