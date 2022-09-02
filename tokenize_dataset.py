import tensorflow as tf
import numpy as np
import torch
import random
import pandas as pd
import os, sys
import time
import datetime
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import argparse
from torch.utils.data import TensorDataset

parser = argparse.ArgumentParser()
parser.add_argument("-model_type", required=True)
parser.add_argument("-data_path", required=True)
parser.add_argument("-female_attributes_path", required=True)
parser.add_argument("-male_attributes_path", required=True)
parser.add_argument("-sequence_length", type=int, default=100)
parser.add_argument("-save_tokenized_data_path", default="tokenized_data.pt")

args = parser.parse_args()
config = vars(args)
print(config)


model_type = args.model_type
data_path = args.data_path
female_attributes_path = args.female_attributes_path
male_attributes_path = args.male_attributes_path
save_tokenized_data_path = args.save_tokenized_data_path
seq_len = args.sequence_length

# random.seed(42)
torch.manual_seed(42) 
if torch.cuda.is_available():
  device = torch.device("cuda")
  print('There are %d GPU(s) available.' % torch.cuda.device_count())
  print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
  print('No GPU available, using the CPU instead.')
  device = torch.device("cpu")

    
# load data
data_all = open(data_path, "r") 
data = data_all.read()
data = np.array(data.split("\n"))
print(data.shape)

# female_attribute_list
female_attribute = open(female_attributes_path, "r") 
f_data = female_attribute.read()
female_attribute_list = f_data.split("\n")
# print(female_attribute_list)
female_attribute.close()


# male_attribute_list
male_attribute = open(male_attributes_path, "r")
m_data = male_attribute.read()
male_attribute_list = m_data.split("\n")
# print(male_attribute_list)
male_attribute.close()


# import tokenizer
if model_type=="albert-large":
    from transformers import AlbertTokenizer
    tokenizer = AlbertTokenizer.from_pretrained('albert-large-v2', do_lower_case=True)
if model_type=="bert-large":
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)


# get vocab list
vocabs = tokenizer.get_vocab()
    

# function to get ids of attribute words. Out of vocabulary words are skipped to avoid wordpiece tokenization.
def get_attributes_ids(gender_attributes):
    gender_attributes_ids = []
    for j in gender_attributes:
        wordpieces = tokenizer(j).input_ids
        if len(wordpieces)==3:
            gender_attributes_ids.append(wordpieces[1])
    return gender_attributes_ids
        
        
        
def tokenize(data):
    # Load the BERT tokenizer.
    print('Loading tokenizer...')

    # Training Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    stereotype_token_id = []
    stereotype_token_index = []
    gender_label = []
    attention_masks = []
    
    
    # get male and female stereopyes
    train_female_attributes_ids = get_attributes_ids(female_attribute_list[:int(0.8*len(female_attribute_list))])
    train_male_attributes_ids = get_attributes_ids(male_attribute_list[:int(0.8*len(male_attribute_list))])
    print("female_attributes_ids: ", train_female_attributes_ids, "\n")
    print("male_attributes_ids: ", train_male_attributes_ids)
    
    # For every sentence...
    for k, sent in enumerate(data):
        encoded_dict = tokenizer.encode_plus(
                            str(sent),                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = seq_len,           # Pad & truncate all sentences.
                            truncation=True,
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )
        
        
        tokens_ids=encoded_dict['input_ids'][0].cpu().numpy().copy()
#         target_output = tokens.copy()

        for i, token in enumerate(tokens_ids):
            if token in train_female_attributes_ids:
                tokens_tensor = torch.tensor([tokens_ids])
                input_ids.append(tokens_tensor)
                attention_masks.append(encoded_dict['attention_mask'])
                stereotype_token_id.append(token)
                stereotype_token_index.append(i)
                gender_label.append(0)
            elif token in train_male_attributes_ids:
                tokens_tensor = torch.tensor([tokens_ids])
                input_ids.append(tokens_tensor)
                attention_masks.append(encoded_dict['attention_mask'])
                stereotype_token_id.append(token)
                stereotype_token_index.append(i)
                gender_label.append(1)

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
#     print(input_ids)
    attention_masks = torch.cat(attention_masks, dim=0)
#     print(attention_masks)
    stereotype_token_id = torch.tensor(stereotype_token_id)
    
    stereotype_token_index = np.array(stereotype_token_index)
    b = np.zeros((stereotype_token_index.size, seq_len))
    b[np.arange(stereotype_token_index.size),stereotype_token_index] = 1
    stereotype_token_index = torch.tensor(b.astype(int))
#     print(stereotype_token_index)
    
    gender_label = torch.tensor(gender_label)

    return input_ids, attention_masks,  stereotype_token_id, stereotype_token_index, gender_label
    
    
def main():    
    input_ids, attention_masks,  stereotype_token_id, stereotype_token_index, gender_label = tokenize(data)


    print('Original: ', data[0])
    print('input_ids: ', input_ids[0], "\n")
    print('gender_label:', gender_label[0], "\n")
    print('stereotype_token_id: ', stereotype_token_id[0], "\n")
    print('stereotype_token_index:', stereotype_token_index[0], "\n")
    print('gender_label:', gender_label[0], "\n")

    tokenized_dataset = TensorDataset(input_ids, attention_masks,  stereotype_token_id, stereotype_token_index, gender_label)
    print('{:>5,} data samples'.format(len(tokenized_dataset)))
    torch.save(tokenized_dataset, save_tokenized_data_path)
    
if __name__ == '__main__':
    main()
