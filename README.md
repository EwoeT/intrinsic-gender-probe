# intrinsic_gender_probe

## 1. Data prep
- We use code from https://github.com/kanekomasahiro/context-debias/blob/main/script/preprocess.sh to extract senteces containing attribute abd stereotype words. <br/>
- Extracted sentences are available https://drive.google.com/drive/folders/1dT62-X2mDBjpe8yvd6flJizeF8mcbPpN?usp=sharing
## 2. Data tokenization
- Tokenize sentences; positional indices of attribute/target words are kept for embedding extraction.  Use attribute for training the detector and stereotypes to test for bias only<br/>
- Args: <br/>
-model_type: type of model "albert-large" or "bert-large <br/>
-data_types: **"attributes"** for generating tokens for attributes and **"stereotypes"** for generating tokens for stereotypes. <br/>
-data_path: path to data <br/>
-save_tokenized_data_path: path to save tokenized data <br/>
-female_list_path: path to female list (attributes or stereotypes) <br/>
-male_list_path: path to male attributes (attributes or stereotypes) <br/>
-all_attributes_and_names_path: path to file containing female and male attributes and names to exclude from sentences containing stereotypes (optional). Removes gender from the context of stereotypes
-sequence_length: max number of tokens to generate per sentence (optional, default: 4) <br/>

- Example:
```
python tokenize_dataset.py \
-model_type "bert-large" \
-data_path "all_attribute__sentences.txt \ #all_stereotype_sentences.txt" (for stereotype sentences)
-data_type "stereotypes" \
-female_list_path 'data/female_attributes.txt' \
-male_list_path 'data/male_attributes.txt' \
-all_attributes_and_names_path 'all_attributes_and_names.txt''
```

## 3. Train gender detector
- Run gender_attribute_classifier.py to train embedding gender detector.  <br/>
- Args: <br/>
-model_type: type of model "albert-large" or "bert-large <br/>
-model_path: path to model <br/>
-save_model_path: path to save model <br/>
-train_data_path: path to train tokenized data <br/>
-val_data_path: path to val tokenized data  <br/>
-epochs: number of epochs (optional, default: 4) <br/>
-batch_size: batch_size (optional, default: 32) <br/>

- Example:
```
python gender_attribute_classifier.py \
-model_type "albert-large" \
-model_path "albert-cda/pytorch_model.bin" \
-epochs 4 \
-train_data_path 'train_attributes_datasets_seed_42_albert_large.pt' \
-val_data_path 'test_attributes_datasets_seed_42_albert_large.pt'
```
## 4. Evaluate on stereotypes
- Run this script with stereotyped tokenized data to evaluate bias in embeddings <br/>
- Args: <br/>
- -model_path: path to trained gender detection model <br/>
- -data_path: path to tokenized data for evaluating stereotypes <br/>

```
python Stereotyping_bias_test.py \
-model_path "gender_classifier.pth" \
-evaluation_data_path 'stereotype_dataset.pt'
```

# Requirements
torch==1.12.1 <br/>
transformers==4.21.2 <br/>
numpy<br/>



