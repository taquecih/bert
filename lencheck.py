#!/usr/bin/env python
# # coding: utf8
# 要素数があってるかチェック

n = 0
cat = ["B-OG", "B-UG", "B-MT", "B-GM", "B-LC"]

#タグ名指定
b = cat[n]
f3n = "tags_" + b + ".txt"

labels = open(f3n, 'r', encoding="utf-8").readlines()
l = [0] * len(labels)

for i in range(len(labels)):
    labels[i] = labels[i].replace("\n","").split(" ")
    l[i] = str(len(labels[i]))
    print(l[i])

l = "\n".join(l)
f3 = open("len.txt", 'w', encoding="utf-8")
f3.write(str(l))
f3.close()