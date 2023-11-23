from text_module.tf_idf import IDFVectorizer
from data_utils.vocab import create_vocab
from text_module.text_embedding import Text_Embedding
from text_module.count_vectorizer import CountVectorizer

def build_text_embedding(config,max_len=None):
    if config['text_embedding']['type']=='pretrained':
        return Text_Embedding(config,max_len)
    if config['text_embedding']['type']=='tf_idf':
        vocab,word_count=create_vocab(config)
        return IDFVectorizer(config["text_embedding"]["d_model"],vocab,word_count)
    if config['text_embedding']['type']=='count_vec':
        vocab,word_count=create_vocab(config)
        return CountVectorizer(config["text_embedding"]["d_model"],vocab)
