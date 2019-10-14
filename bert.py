#!/usr/bin/env python
# # coding: utf8

# Load The Data

import pandas as pd
import numpy as np
from tqdm import tqdm, trange

# data = pd.read_csv("lab_test.csv", encoding="utf-8").fillna(method="ffill")
# # print(data.head(10))
# # print(data.tail(10))
#
# class SentenceGetter(object):
#
#     def __init__(self, data):
#         self.n_sent = 1
#         self.data = data
#         self.empty = False
#         agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(), s["Tag"].values.tolist())]
#         self.grouped = self.data.groupby("Sentence #").apply(agg_func)
#         self.sentences = [s for s in self.grouped]
#
#     def get_next(self):
#         try:
#             s = self.grouped["Sentence: {}".format(self.n_sent)]
#             self.n_sent += 1
#             return s
#         except:
#             return None
#
# getter = SentenceGetter(data)
#
# sentences = [" ".join([s[0] for s in sent]) for sent in getter.sentences]
# print(sentences[0])
#
# labels = [[s[1] for s in sent] for sent in getter.sentences]
# print(labels[0])

n = 4
cat = ["B-OG", "B-UG", "B-MT", "B-GM", "B-LC"]

#タグ名指定
b = cat[n]
f2n = "sent_" + b + "_os.txt"
f3n = "tags_" + b + "_os.txt"

sentences = open(f2n, 'r', encoding="utf-8").readlines()
labels = open(f3n, 'r', encoding="utf-8").readlines()

val_len = 250
taglist = []

for i in range(len(labels)):
    labels[i] = labels[i].replace("\n","").split(" ")
    taglist = taglist + labels[i]

tags_vals = list(set(taglist))
tag2idx = {t: i for i, t in enumerate(tags_vals)}
print(tag2idx)

val_sentences = sentences[-1 * val_len:-1]
val_labels = labels[-1 * val_len:-1]
tr_sentences = sentences[0:len(sentences) - val_len - 1]
tr_labels = labels[0:len(sentences) - val_len - 1]

# Apply Bert

import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam

MAX_LEN = 128
bs = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

torch.cuda.get_device_name(0) 

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)

tr_tokenized_texts = [tokenizer.tokenize(tr_sent) for tr_sent in tr_sentences]
val_tokenized_texts = [tokenizer.tokenize(val_sent) for val_sent in val_sentences]

tr_input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(tr_txt) for tr_txt in tr_tokenized_texts], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
val_input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(val_txt) for val_txt in val_tokenized_texts], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

tr_tags = pad_sequences([[tag2idx.get(l) for l in tr_lab] for tr_lab in tr_labels], maxlen=MAX_LEN, value=tag2idx["O"], padding="post", dtype="long", truncating="post")
val_tags = pad_sequences([[tag2idx.get(l) for l in val_lab] for val_lab in val_labels], maxlen=MAX_LEN, value=tag2idx["O"], padding="post", dtype="long", truncating="post")

tr_attention_masks = [[float(i>0) for i in tr_ii] for tr_ii in tr_input_ids]
val_attention_masks = [[float(i>0) for i in val_ii] for val_ii in val_input_ids]

# tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags, random_state=2018, test_size=0.1)
# tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2018, test_size=0.1)

tr_inputs = torch.tensor(tr_input_ids)
val_inputs = torch.tensor(val_input_ids)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_attention_masks)
val_masks = torch.tensor(val_attention_masks)

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2idx))
model.cuda();

FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters()) 
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

from seqeval.metrics import f1_score

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

epochs = 50
max_grad_norm = 1.0

for _ in trange(epochs, desc="Epoch"):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # forward pass
        loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        # backward pass
        loss.backward()
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        model.zero_grad()
    # print train loss per epoch
    print("Train loss: {}".format(tr_loss/nb_tr_steps))
    # VALIDATION on validation set
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        
        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss/nb_eval_steps
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
    valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
    print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))

model.eval()
predictions = []
true_labels = []
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
for batch in valid_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        tmp_eval_loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        
    logits = logits.detach().cpu().numpy()
    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    label_ids = b_labels.to('cpu').numpy()
    true_labels.append(label_ids)
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)

    eval_loss += tmp_eval_loss.mean().item()
    eval_accuracy += tmp_eval_accuracy

    nb_eval_examples += b_input_ids.size(0)
    nb_eval_steps += 1

pred_tags = [[tags_vals[p_i] for p_i in p] for p in predictions]
valid_tags = [[tags_vals[l_ii] for l_ii in l_i] for l in true_labels for l_i in l ]
print("Validation loss: {}".format(eval_loss/nb_eval_steps))
print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
print(pred_tags)
print(valid_tags)

