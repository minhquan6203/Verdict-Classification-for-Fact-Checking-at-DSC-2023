import os
import logging
from typing import Text, Dict, List
import pandas as pd
import torch
import transformers
from model.init_model import get_model
from data_utils.load_data import Get_Loader
from tqdm import tqdm
from data_utils.load_data import create_ans_space

class Predict:
    def __init__(self, config: Dict):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.answer_space=create_ans_space(config)
        self.checkpoint_path=os.path.join(config["train"]["output_dir"], "best_model.pth")
        self.model = get_model(config, num_labels = len(self.answer_space))
        self.dataloader = Get_Loader(config)
    def predict_submission(self):
        transformers.logging.set_verbosity_error()
        logging.basicConfig(level=logging.INFO)
    
        # Load the model
        logging.info("Loading the best model...")
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        # Obtain the prediction from the model
        logging.info("Obtaining predictions...")
        test =self.dataloader.load_test()
        submits=[]
        ids=[]
        self.model.eval()
        with torch.no_grad():
            for it, (sent1, sent2, id) in enumerate(tqdm(test)):
                with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=True):
                    logits = self.model(sent1,sent2)
                preds = logits.argmax(axis=-1).cpu().numpy()
                answers = [self.answer_space[i] for i in preds]
                submits.extend(answers)
                if isinstance(id, torch.Tensor):
                    ids.extend(id.tolist())
                else:
                    ids.extend(id)
                    
        data = {'id': ids,'label': submits }
        df = pd.DataFrame(data)
        df.to_csv('./submission.csv', index=False)