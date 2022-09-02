# intrinsic_gender_probe
```
!python gender_attribute_classifier.py \
-model_type "albert-large" \
-model_path "../../CDA/counterfactual-data-substitution-master/zari-albert-cda/pytorch_model.bin" \
-epochs 4 -train_data_path '../bias_in_bios/bias_in_bios_classifier_fine_tuned_embeddings/intrinsic_bias/datasets/albert_large/train_attributes_datasets_seed_42_albert_large.pt' \
-val_data_path '../bias_in_bios/bias_in_bios_classifier_fine_tuned_embeddings/intrinsic_bias/datasets/albert_large/test_attributes_datasets_seed_42_albert_large.pt'
```
Bert model is adapted from huggingface https://huggingface.co/transformers/model_doc/bert.html
Bert fine-tuning codes are adapted from:https://mccormickml.com/2019/07/22/BERT-fine-tuning/

- 1_bias_class_discriminator.ipynb: To train classifier for bias detection
- 2_bias_classification_straight_through.ipynb: To trian classifier with straight through technique
- 3_latent_embedding_classifier.ipynb: Trains classifier to detect if latent encoding is biased or neutral (in the case of bias mitigation) / male or female (in the case of gender obfuscation)
- 4_generate_neutral_latent_representation.ipynb: Generates disentangled (neutral) latent representation
- 5_bias_mitigation_MLM.ipynb: Main style transfer code
- 6_TEST_bias_mitigation_MLM.ipynb: Code for evaluation

# Requirements
transformers==4.10.0
