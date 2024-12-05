import os
import numpy as np
import torch
from torch.utils.data import Dataset

class EmotionDataset_MultiTask(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.sample_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir)
                            if os.path.isdir(os.path.join(data_dir, d))]
        self.transform = transform

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        landmarks_seq = np.load(os.path.join(sample_dir, 'landmarks.npy'))  # (seq_len, input_dim1)
        lbp_seq = np.load(os.path.join(sample_dir, 'lbp.npy'))              # (seq_len, input_dim2)
        
        combined_seq = np.concatenate((landmarks_seq, lbp_seq), axis=1)     # (seq_len, input_dim1 + input_dim2)
        combined_seq = torch.tensor(combined_seq, dtype=torch.float32)
        
        if self.transform:
            combined_seq = self.transform(combined_seq)
        
        valence = np.load(os.path.join(sample_dir, 'Valence.npy'))        # (seq_len,)
        arousal = np.load(os.path.join(sample_dir, 'Arousal.npy'))        # (seq_len,)
        liking = np.load(os.path.join(sample_dir, 'Liking.npy'))          # (seq_len,)
        
        valence = torch.tensor(valence[-1], dtype=torch.float32)
        arousal = torch.tensor(arousal[-1], dtype=torch.float32)
        liking = torch.tensor(liking[-1], dtype=torch.float32)
        
        return combined_seq, valence, arousal, liking
