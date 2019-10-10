#!/usr/bin/env python
# # coding: utf8

import random

n = 4
cat = ["B-OG", "B-UG", "B-MT", "B-GM", "B-LC"]

#タグ名指定
b = cat[n]
f2n = "sent_" + b + ".txt"
f3n = "tags_" + b + ".txt"

sentences = open(f2n, 'r', encoding="utf-8").readlines()
labels = open(f3n, 'r', encoding="utf-8").readlines()

val_len = 250

val_sentences = sentences[-1 * val_len:-1]
val_labels = labels[-1 * val_len:-1]
sentences = sentences[0:len(sentences) - val_len - 1]
labels = labels[0:len(sentences) - val_len - 1]
labeled = []

for i in range(len(labels)):
    ls = labels[i].replace("\n","").split(" ")
    if b in ls:
        labeled.append(i)

aug = len(sentences) - len(labeled)
smpnums = random.choices(labeled, k=aug)

for smpnum in smpnums:
    sentences.append(sentences[smpnum])
    labels.append(labels[smpnum])

for i in range(val_len - 1):
    sentences.append(val_sentences[i])
    labels.append(val_labels[i])

s = "".join(sentences)
l = "".join(labels)

f2sn = "sent_" + b + "_os.txt"
f3sn = "tags_" + b + "_os.txt"

f2s = open(f2sn, 'w', encoding="utf-8")
f2s.write(s)
f2s.close()

f3s = open(f3sn, 'w', encoding="utf-8")
f3s.write(l)
f3s.close()
