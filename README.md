# intrinsic_gender_probe

1. Data prep
2. Data tokenization
3. Train gender detector: Run gender_attribute_classifier.py to train embedding gender detector. Args: 
-model_type: type of model "albert-large" or "bert-large
- model_path: path to model
- save_model_path: path to save model
- train_data_path: path to train tokenized data
- val_data_path: path to val tokenized data 
- epochs: number of epochs (optional, default: 4)
- batch_size: batch_size (optional, default: 32)
```
4. Evaluate on stereotypes
!python gender_attribute_classifier.py \
-model_type "albert-large" \
-model_path "albert-cda/pytorch_model.bin" \
-epochs 4 \
-train_data_path 'albert_large/train_attributes_datasets_seed_42_albert_large.pt' \
-val_data_path 'albert_large/test_attributes_datasets_seed_42_albert_large.pt'
```


# Requirements
torch==1.12.1 <br/>
transformers==4.21.2 <br/>
numpy<br/>



