import torch
import pandas as pd
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F                                                               
from tqdm import trange
from tqdm import tqdm
from functools import partial
from glob import glob
from time import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
import numpy

class DB(Dataset):
    #def __init__(self, filename, indices, emb_dim = 64, precision=32):
    def __init__(self, filename, emb_dim = 8, precision=32):
        #from torch.nn.utils.rnn import pad_sequence

        if precision==16:
            self.precision = torch.half
        elif precision==32:
            self.precision = torch.float32
        elif precision==64:
            self.precision = torch.float64
        self.filename = filename
        st = time()
        f=pd.read_csv(self.filename)
        print(time()-st, 'loading csv file')
        st = time()
        #self.indices = indices
        
        self.num_mol = len(f['smi'])
        self.smi = f['smi']	
        word_set = set(list((''.join(self.smi))))
	
        vocab = {tkn: i+2 for i, tkn in enumerate(word_set)}
        vocab['<unk>'] = 0
        vocab['<pad>'] = 1
	
        embedding_layer = nn.Embedding(num_embeddings=len(vocab), 
	                               embedding_dim= emb_dim,
	                               padding_idx = 1)
        print(self.num_mol) 
        inp = []
        for idx in range(len(self.smi)):
            encoded = [vocab[i] for i in self.smi[idx]]
            idxes = torch.LongTensor(encoded)
            lookup_result = embedding_layer.weight[idxes, :]
            inp.append(lookup_result)
        #print(f['s_energy(ev)'])
        #print(torch.from_numpy(np.array(f['s_energy(ev)'])))
        #exit(-1)
        self.s_inp = np.array(f['s_energy(ev)'])
        self.t_inp = np.array(f['t_energy(ev)'])

        self.inp = pad_sequence(inp, batch_first=True, padding_value = 0) # (batch_size, len(smiles), embedding_size)
	
    def __len__(self):
        return self.num_mol
    def __getitem__(self, idx):
        self.mol_idx = [ i for i in range(self.num_mol) ]
        smi_inp = self.inp[idx,:]
        singlet_energy =(self.s_inp[idx])
        triplet_energy =(self.t_inp[idx])

        return (smi_inp).type(self.precision),\
               torch.as_tensor((singlet_energy)).type(self.precision),\
               torch.as_tensor((triplet_energy)).type(self.precision)
