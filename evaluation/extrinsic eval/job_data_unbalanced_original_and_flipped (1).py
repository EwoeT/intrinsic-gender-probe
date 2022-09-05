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
from sklearn.utils import shuffle
# import job_data


df = pd.read_csv("../data_prep/preprocessed_flipped.csv")

df_job = pd.read_csv("../data_prep/occupation.csv", header=None)

for i in range(df["label"].nunique()):
    df.loc[df['label'] == i, 'prof'] = df_job.iloc[i][1]

gen_dist = {}
for i in range(df["label"].nunique()):
    gen_frac = (len(df[(df["label"]==i) & (df["gender"]=="F")]))/(len(df[df["label"]==i]))
    gen_dist[df_job.iloc[i][1]] = gen_frac
gen_dist

gen_dist = {}
for i in range(df["label"].nunique()):
    gen_frac = (len(df[(df["label"]==i) & (df["gender"]=="F")]))/(len(df[df["label"]==i]))
    gen_dist[df_job.iloc[i][1]] = gen_frac
gen_dist

gen_dist = dict(sorted(gen_dist.items(), key=lambda item: item[1]))
gen_dist

job_list = list(gen_dist.keys())
male_jobs = job_list[:7]
female_jobs = job_list[-7:]


processed_bios_df = df[["raw_title", "gender", "label", "scrubbed", "bio", "flipped_bio", "swapped", "prof"]]


gendered_jobs = female_jobs + male_jobs


female_bios_df = processed_bios_df.loc[processed_bios_df['prof'].isin(female_jobs)]
female_bios_df["job_cat"] = "female_job"
female_bios_df["job_cat_id"] = int(0)
male_bios_df = processed_bios_df.loc[processed_bios_df['prof'].isin(male_jobs)]
male_bios_df["job_cat"] = "male_job"
male_bios_df["job_cat_id"] = int(1)

final_bios_df= pd.concat([female_bios_df, male_bios_df])
final_bios_df = shuffle(final_bios_df, random_state=42)