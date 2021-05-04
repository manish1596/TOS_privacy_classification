#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/prateekjoshi565/Fine-Tuning-BERT/blob/master/Fine_Tuning_BERT_for_Spam_Classification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Install Transformers Library

# In[ ]:


get_ipython().system('pip install transformers')


# In[38]:


import numpy as np
import json
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import BertModel, BertTokenizerFast

# specify GPU
device = torch.device("cpu")


# # Load Dataset

# In[2]:


df = pd.read_csv("test_tos.csv")
print(df.columns)
df.head()


# In[3]:


df.shape


# In[4]:


# check class distribution
df['label'].value_counts(normalize = True)


# # Split train dataset into train, validation and test sets

# In[6]:


train_text, temp_text, train_labels, temp_labels = train_test_split(df['text'], df['label'], 
                                                                    random_state=2018, 
                                                                    test_size=0.99, 
                                                                    stratify=df['label'])

# we will use temp_text and temp_labels to create validation and test set
val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, 
                                                                random_state=2018, 
                                                                test_size=0.99, 
                                                                stratify=temp_labels)


# # Import BERT Model and BERT Tokenizer

# In[7]:


# import BERT-base pretrained model
bert = BertModel.from_pretrained('bert-base-uncased')

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


# In[8]:


# sample data
text = ["this is a bert model tutorial", "we will fine-tune a bert model"]

# encode text
sent_id = tokenizer.batch_encode_plus(text, padding=True, return_token_type_ids=False)


# In[9]:


# output
print(sent_id)


# # Tokenization

# In[10]:


# get length of all the messages in the train set
seq_len = [len(i.split()) for i in train_text]

pd.Series(seq_len).hist(bins = 30)


# In[11]:


max_seq_len = 25


# In[12]:


# tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)


# # Convert Integer Sequences to Tensors

# In[13]:


# for train set
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

# for validation set
val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

# for test set
test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())


# # Create DataLoaders

# In[18]:


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#define a batch size
batch_size = 32

# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)

# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)


# # Freeze BERT Parameters

# In[19]:


# freeze all the parameters
for param in bert.parameters():
    param.requires_grad = False


# # Define Model Architecture

# In[20]:


class BERT_Arch(nn.Module):

    def __init__(self, bert):
      
        super(BERT_Arch, self).__init__()

        self.bert = bert 

        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu =  nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768,512)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512,3)

        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):

        #pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)

        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)

        # apply softmax activation
        x = self.softmax(x)

        return x


# In[21]:


# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)

# push the model to GPU
model = model.to(device)


# In[22]:


# optimizer from hugging face transformers
from transformers import AdamW

# define the optimizer
optimizer = AdamW(model.parameters(), lr = 1e-3)


# # Find Class Weights

# In[23]:


from sklearn.utils.class_weight import compute_class_weight

#compute the class weights
class_wts = compute_class_weight('balanced', np.unique(train_labels), train_labels)

print(class_wts)


# In[24]:


# convert class weights to tensor
weights= torch.tensor(class_wts,dtype=torch.float)
weights = weights.to(device)

# loss function
cross_entropy  = nn.NLLLoss(weight=weights) 

# number of training epochs
epochs = 10


# # Load Saved Model

# In[25]:


#load weights of best model
path = 'saved_weights.pt'
model.load_state_dict(torch.load(path))


# # Get Predictions for Test Data

# In[29]:


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

# get predictions for test data
with torch.no_grad():
    preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()

total_score=0.0
max_score_possible=0.0
for pred in preds:
    curr_scores=softmax(pred)
    total_score += curr_scores[0]+2*curr_scores[1]+3*curr_scores[2]
    max_score_possible += 3.0

final_privacy_grade="A"
if(total_score < max_score_possible/5):
    final_privacy_grade="E"
if(total_score >= max_score_possible/5 and total_score < 2*max_score_possible/5):
    final_privacy_grade="D"
if(total_score >= 2*max_score_possible/5 and total_score < 3*max_score_possible/5):
    final_privacy_grade="C"
if(total_score >= 3*max_score_possible/5 and total_score < 4*max_score_possible/5):
    final_privacy_grade="B"
if(total_score >= 4*max_score_possible/5 and total_score <= 5*max_score_possible/5):
    final_privacy_grade="A"


# In[56]:


final_output={}
final_output["tos_document"]="input.txt"
final_output["privacy_grade"]=final_privacy_grade
final_output["points"]=[]
for i,curr_tuple in enumerate([(n,s) for (n,s) in test_text.items()]):
    #print(i)
    curr_dict={}
    curr_dict["quoteText"]=curr_tuple[1]
    if(np.argmax(preds[i])==0):
        curr_dict["verdict"]="bad"
    if(np.argmax(preds[i])==1):
        curr_dict["verdict"]="neutral"
    if(np.argmax(preds[i])==2):
        curr_dict["verdict"]="good"
    final_output["points"].append(curr_dict)
output_file=open("final_output.json","w")
json.dump(final_output, output_file)

