# intrinsic_gender_probe
```
!python gender_attribute_classifier.py \
-model_type "albert-large" \
-model_path "albert-cda/pytorch_model.bin" \
-epochs 4 -train_data_path 'albert_large/train_attributes_datasets_seed_42_albert_large.pt' \
-val_data_path 'albert_large/test_attributes_datasets_seed_42_albert_large.pt'
```


# Requirements
torch==1.12.1 <br/>
transformers==4.21.2 <br/>
numpy<br/>



