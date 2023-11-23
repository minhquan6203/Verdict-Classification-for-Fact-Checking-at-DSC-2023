from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.init_text_embedding import build_text_embedding
from encoder_module.init_encoder import build_multi_modal_encoder


class Trans_Model_New(nn.Module):
    def __init__(self,config: Dict, num_labels: int):
     
        super(Trans_Model_New, self).__init__()
        self.num_labels = num_labels
        self.intermediate_dims = config["model"]["intermediate_dims"]
        self.dropout=config["model"]["dropout"]
        self.type_text_embedding=config["text_embedding"]['type']
        self.text_embbeding1 = build_text_embedding(config,64)
        self.text_embbeding2 = build_text_embedding(config,32)  
        self.fusion = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.multi_encoder = build_multi_modal_encoder(config)
        self.classifier = nn.Linear(self.intermediate_dims, self.num_labels)
        self.attention_weights = nn.Linear(self.intermediate_dims, 1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, id1_text: List[str], id2_text: List[str], labels: Optional[torch.LongTensor] = None):

        embbed_col1, mask_col1= self.text_embbeding1(id1_text)
        embbed_col2, mask_col2 = self.text_embbeding2(id2_text)
            
        encoded_feature1, encoded_feature2 = self.multi_encoder(embbed_col1, mask_col1, embbed_col2, mask_col2)
        fused_feature = self.fusion(torch.cat([encoded_feature1, encoded_feature2], dim=1))

        fused_attended = self.attention_weights(torch.tanh(fused_feature))
        
        attention_weights = torch.softmax(fused_attended, dim=1)
        
        fused_output = torch.sum(attention_weights * fused_feature, dim=1)

        logits = self.classifier(fused_output)
        logits = F.log_softmax(logits, dim=-1)
        if labels is not None:
            loss = self.criterion(logits, labels)
            return logits,loss
        else:
            return logits

def createTrans_Model_New(config: Dict, answer_space: List[str]) ->Trans_Model_New:
    return Trans_Model_New(config, num_labels=len(answer_space))