import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from nltk.tokenize import word_tokenize

import numpy as np
from models import models
import argparse

def _open_dict(dict_path):
    dico_file = open(dict_path).read().splitlines()
    dico = {word.strip(): idx for idx, word in enumerate(dico_file)}
    return dico


def encode_sentence(sent, model, embed, dico):
    sent = word_tokenize(sent)
    sent_in = torch.Tensor(1, len(sent), 620)
    for i, w in enumerate(sent):
        if w in dico:
            sent_in[0, i] = torch.from_numpy(embed[dico[w]])
        else:
            sent_in[0, i] = torch.from_numpy(embed[dico["UNK"]])
    return model(sent_in)[0][-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('text', help="Text to encode")
    parser.add_argument('--model_path', help="Path of the trained model", default="/data/best_model.pth.tar")
    parser.add_argument('--utable', help="Path to the utable file", default="/data/utable.npy")
    parser.add_argument('--dictionary', help='Path to dictionary file', default='/data/dictionary.txt')
    args = parser.parse_args()


    model_state = torch.load(args.model_path)
    model = models.joint_embedding()
    model.load_state_dict(model_state['state_dict'])
    model = model.eval()
    embed = np.load(args.utable, encoding='latin1')
    
    dico = _open_dict(args.dictionary)
    print(encode_sentence(args.text, model.cap_emb,embed, dico))
    
