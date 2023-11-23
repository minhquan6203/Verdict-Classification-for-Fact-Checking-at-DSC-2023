from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Optional
import pandas as pd
import os
import numpy as np
import re
def preprocess_text(text: str) -> str:    
    text = re.sub(r"['\",\?:\-!,\;]", "", text)
    text = text.strip()
    text = " ".join(text.split())
    text = text.replace('\n',' ').strip().lower()
    return text

class CustomDataset(Dataset):
    def __init__(self, data, with_labels=True):
        self.data = data  # pandas dataframe
        self.with_labels = with_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        idx=self.data.loc[index, 'idx']
        sent1 = preprocess_text(str(self.data.loc[index, 'id1_text']))
        sent2 = preprocess_text(str(self.data.loc[index, 'id2_text']))
        #nếu dùng id:
        # sent1 =str(self.data.loc[index, 'id1'])+' '+str(self.data.loc[index, 'sentence1'])
        # sent2 =str(self.data.loc[index, 'id2'])+' '+str(self.data.loc[index, 'sentence2'])
        if self.with_labels:  # True if the dataset has labels
            labels = self.data.loc[index, 'Label']
            return sent1, sent2, labels, idx
        else:
            return sent1, sent2, idx
        
class Get_Loader:
    def __init__(self, config):
        self.train_path=os.path.join(config['data']['dataset_folder'],config['data']['train_dataset'])
        self.train_batch=config['train']['per_device_train_batch_size']

        self.val_path=os.path.join(config['data']['dataset_folder'],config['data']['val_dataset'])
        self.val_batch=config['train']['per_device_valid_batch_size']

        self.test_path=os.path.join(config['inference']['test_dataset'])
        self.test_batch=config['inference']['batch_size']

    def load_train_dev(self):
        train_df=pd.read_csv(self.train_path)
        answer_space = list(np.unique(train_df['Label']))
        # answer_space = ['SUPPORTED','NEI','REFUTED']
        label_to_index = {label: index for index, label in enumerate(answer_space)}
        train_df['Label'] = train_df['Label'].map(label_to_index)

        val_df=pd.read_csv(self.val_path)
        val_df['Label'] = val_df['Label'].map(label_to_index)
        print("Reading training data...")
        train_set = CustomDataset(train_df)
        print("Reading validation data...")
        val_set = CustomDataset(val_df)
    
        train_loader = DataLoader(train_set, batch_size=self.train_batch, num_workers=2,shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.val_batch, num_workers=2,shuffle=True)
        return train_loader, val_loader
    
    def load_test(self):
        test_df=pd.read_csv(self.test_path)
        print("Reading testing data...")
        test_set = CustomDataset(test_df,with_labels=False)
        test_loader = DataLoader(test_set, batch_size=self.test_batch, num_workers=2, shuffle=False)
        return test_loader
    
def create_ans_space(config: Dict):
    train_path=os.path.join(config['data']['dataset_folder'],config['data']['train_dataset'])
    train_df=pd.read_csv(train_path)
    answer_space = list(np.unique(train_df['Label']))
    # answer_space = ['SUPPORTED','NEI','REFUTED']
    return answer_space