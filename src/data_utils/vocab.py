from typing import List, Dict, Optional
from datasets import load_dataset
import os

def create_vocab(config: Dict):
    data_folder=config['data']['dataset_folder']
    train_set=config["data"]["train_dataset"]
    val_set=config["data"]["val_dataset"]
    test_set=config["data"]["test_dataset"]
    dataset = load_dataset(
        "csv", 
        data_files={
            "train": os.path.join(data_folder, train_set),
            "val": os.path.join(data_folder, val_set),
            #"test": os.path.join(data_folder, test_set)
        }
    )

    word_counts = {}

    for data_file in dataset.values():
        try:
            for item in data_file['id1_text']:
                for word in item.split():
                    # word=word.lower()
                    if word not in word_counts:
                        word_counts[word] = 1
                    else:
                        word_counts[word] += 1
            for item in data_file['id2_text']:
                for word in item.split():
                    # word=word.lower()
                    if word not in word_counts:
                        word_counts[word] = 1
                    else:
                        word_counts[word] += 1
        except:
            pass

    sorted_word_counts = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True))
    vocab = list(sorted_word_counts.keys())

    # Thêm từ "[unknown]" vào vocab
    vocab.append("[unknown]")

    return vocab, sorted_word_counts
