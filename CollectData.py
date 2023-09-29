import gdown

file_id = ['1UEjb_mHXBpCbysi3-6x-WS-P1P6zFtIN','1bh1N6I5XbpQystqngzqm0TQUJeAGah4L','1zRwN2U0pTGKFuZCLDlJ-Aeqi2bnFmuGD','1E8PeZ_ix5xX_O3tm-E29ecqNtZsdout-']
output = ['./base_model_google.zip', './tokenizer_pretrained_google.zip', './data_google.zip','./small_hf_google.zip']

for idx in range(len(file_id)):
    gdown.download(id=file_id[idx], output=output[idx], quiet=False)
