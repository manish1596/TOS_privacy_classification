import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
import argparse
from transformers import BertModel, BertTokenizerFast

# specify GPU
device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()

f=open(args.filename)
s=f.read()
para_list=s.strip().split('\n')
#for p in para_list:
    #print(p)
sentence_list=[]
for paragraph in para_list:
    sentence_list.extend(paragraph.split('. '))
original_sentence_list=[]
for s in sentence_list:
    if len(s)>20:
        original_sentence_list.append(s)
f_test_csv=open('test_tos.csv', 'w')
f_test_csv.write("label,text\n")
for s in original_sentence_list:
    f_test_csv.write("0,"+s.replace(',','').replace('\n','')+"\n")
f_test_csv.close()
