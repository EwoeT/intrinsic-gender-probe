import tensorflow as tf
import numpy as np
import torch
import random
import pandas as pd
import os, sys
import time
import datetime
from torch import nn
import argparse
from torch.utils.data import TensorDataset

parser = argparse.ArgumentParser()
parser.add_argument("-model_path", required=True)
parser.add_argument("-evaluation_data_path", required=True)
args = parser.parse_args()
config = vars(args)
print(config)


model_path = args.model_path
evaluation_data_path = args.evaluation_data_path


torch.manual_seed(42) 
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name())
else:
  print('No GPU available, using the CPU instead.')
  device = torch.device("cpu")
    
test_dataset = torch.load(evaluation_data_path)

model = torch.load(model_path)

# import liang_albert_model
# import importlib
# importlib.reload(liang_albert_model)
# AlbertForSequenceClassification = liang_albert_model.AlbertForSequenceClassification
# model = AlbertForSequenceClassification.from_pretrained("albert-large-v2", num_labels = 2)
# model.albert.load_state_dict(torch.load(model_path).albert.state_dict())
# model.classifier.load_state_dict(torch.load(model_path).classifier.state_dict())
model.to(device)


import time
import datetime

# from transformers.optimization import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# import numpy as np


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


import random
from tqdm.notebook import tqdm

def test(model, test_dataloader):
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)



   

    # ========================================
    #               Evaliuating
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running evaluation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        stereotype_token_index = batch[3].to(device)
        b_target_labels = batch[4].to(device)



        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            outputs = model(b_input_ids, 
                                 token_type_ids=None,
                                 stereotype_token_index=stereotype_token_index,
                                 attention_mask=b_input_mask, 
                                 labels=b_target_labels,
                                )

        loss, logits = outputs.loss, outputs.logits
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.reshape(len(b_target_labels),2).detach().cpu().numpy()
        label_ids = b_target_labels.to('cpu').numpy()
#             print(label_ids.shape)

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)
#             print(total_eval_accuracy)


    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
#         print(total_eval_accuracy)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(test_dataloader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

test_dataloader = DataLoader(
        test_dataset, # The validation samples.
        )

test(model, test_dataloader)
time.sleep(4)
