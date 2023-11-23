from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.init_text_embedding import build_text_embedding
from encoder_module.init_encoder import build_multi_modal_encoder

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Linear(config["model"]["intermediate_dims"], config["model"]["intermediate_dims"])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config["model"]["dropout"])
        self.fc2 = nn.Linear(config["model"]["intermediate_dims"], 1)

    def forward(self, features: torch.Tensor):
        output = self.dropout(self.relu(self.fc1(features)))
        output = self.fc2(output)

        return output

class ParallelAttentionTransformer(nn.Module):
    def __init__(self,config: Dict, num_labels: int) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.text_embbeding = build_text_embedding(config)
        self.encoder = build_multi_modal_encoder(config)

        self.text_attr_reduce = MLP(config)
        self.text_proj = nn.Linear(config["model"]["intermediate_dims"], config["model"]["intermediate_dims"])
      
        self.layer_norm = nn.LayerNorm(config["model"]["intermediate_dims"])
        self.criterion = nn.CrossEntropyLoss()
        self.fusion = nn.Sequential(
            nn.Linear(config["model"]["intermediate_dims"] + config["model"]["intermediate_dims"], config["model"]["intermediate_dims"]),
            nn.ReLU(),
            nn.Dropout(config["model"]["dropout"]),
        )
        self.classify = nn.Linear(config["model"]["intermediate_dims"], self.num_labels)

    def forward(self, id1_text: List[str], id2_text: List[str], labels: Optional[torch.LongTensor] = None):

        embbed_col1, mask_col1= self.text_embbeding(id1_text)
        embbed_col2, mask_col2 = self.text_embbeding(id2_text)

        # performing co-attention
        encoded_feature1, encoded_feature2 = self.encoder(embbed_col1, mask_col1, embbed_col2, mask_col2)

        feature1_attended = self.text_attr_reduce(encoded_feature1)
        feature1_attended = F.softmax(feature1_attended, dim=1)

        feature2_attended = self.text_attr_reduce(encoded_feature2)
        feature2_attended = F.softmax(feature2_attended, dim=1)


        weighted_feature1_attended = (encoded_feature1 * feature1_attended).sum(dim=1)
        weighted_feature2_attended = (encoded_feature2 * feature2_attended).sum(dim=1)
        output = self.fusion(torch.cat([weighted_feature1_attended, weighted_feature2_attended], dim=1))
        #output = self.layer_norm(self.text_proj(weighted_feature1_attended) + self.text_proj(weighted_feature2_attended))
        logits = self.classify(output)
        logits = F.log_softmax(logits, dim=-1)

        if labels is not None:
            loss = self.criterion(logits, labels)
            return logits,loss
        else:
            return logits

def createParallelAttentionTransformer(config: Dict, answer_space: List[str]) ->ParallelAttentionTransformer:
    return ParallelAttentionTransformer(config, num_labels=len(answer_space))