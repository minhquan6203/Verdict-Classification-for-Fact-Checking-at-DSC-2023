from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.init_text_embedding import build_text_embedding
from utils.cnn import Text_CNN
from encoder_module.init_encoder import build_uni_modal_encoder


class Text_CNN_Model(nn.Module):
    def __init__(self,config: Dict, num_labels: int):
     
        super(Text_CNN_Model, self).__init__()
        self.num_labels = num_labels
        self.intermediate_dims = config["model"]["intermediate_dims"]
        self.hidden_dim = config["model"]["hidden_dim"]
        self.dropout=config["model"]["dropout"]
        self.d_text = config["text_embedding"]['d_features']
        self.text_embbeding = build_text_embedding(config)
        self.encoder = build_uni_modal_encoder(config)
        self.classifier = Text_CNN(self.intermediate_dims,self.hidden_dim ,self.num_labels)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, id1_text: List[str], id2_text: List[str], labels: Optional[torch.LongTensor] = None):

        embbed, mask = self.text_embbeding(id1_text, id2_text)
        encoded_feature = self.encoder(embbed, mask)
        logits = self.classifier(encoded_feature)
        logits = F.softmax(logits, dim=-1)

        if labels is not None:
            loss = self.criterion(logits, labels)
            return logits,loss
        else:
            return logits

def createText_CNN_Model(config: Dict, answer_space: List[str]) -> Text_CNN_Model:
    return Text_CNN_Model(config, num_labels=len(answer_space))
