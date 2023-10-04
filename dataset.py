import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import argparse

class KoT5SummaryDataset(Dataset):
    def __init__(self, file, tokenizer, max_len, ignore_index=-100):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.docs = pd.read_json(file)
        self.len = self.docs.shape[0]

        self.pad_index = self.tokenizer.pad_token_id
        self.ignore_index = ignore_index
    
    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def __getitem__(self, idx):
        df = self.docs.iloc[idx]
        input_ids = self.tokenizer.encode(df['passage'])
        input_ids = self.add_padding_data(input_ids)
        label_ids = self.tokenizer.encode(df['summary'])
        label_ids.append(self.tokenizer.eos_token_id)
        dec_input_ids = [self.tokenizer.decoder_start_token_id]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids)
        label_ids = self.add_ignored_data(label_ids)

        return {'input_ids': np.array(input_ids, dtype=np.int_),
                'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
                'labels': np.array(label_ids, dtype=np.int_)}
    def __len__(self):
        return self.len
    

class KoT5SummaryModule(L.LightningDataModule):
    def __init__(self, train_file, test_file, tok, max_len=512, batch_size=8, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.max_len = max_len
        self.train_file_path = train_file
        self.test_file_path = test_file
        self.tok = tok
        self.num_workers = num_workers
    
    # no add_model_specific_args
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False
        )
        parser.add_argument("--num_workers", default=4, type=int)
        return parser

    def setup(self, stage):
        self.train = KoT5SummaryDataset(file=self.train_file_path, tokenizer=self.tok, max_len=self.max_len)
        self.test = KoT5SummaryDataset(file=self.test_file_path, tokenizer=self.tok, max_len=self.max_len)
    
    def train_dataloader(self):
        train = DataLoader(self.train,batch_size=self.batch_size,num_workers=self.num_workers, shuffle=True)
        return train
    
    def val_dataloader(self):
        val = DataLoader(self.test,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers, shuffle=False)
        return val

    def test_dataloader(self):
        test = DataLoader(self.test,batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return test