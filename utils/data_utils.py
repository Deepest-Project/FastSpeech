import random
import numpy as np
import hparams
import torch
import torch.utils.data
import os
import pickle as pkl

from text import text_to_sequence


def load_filepaths_and_text(metadata, teacher_path, split="|"):
    filepaths_and_text = []
    with open(metadata, encoding='utf-8') as f:
        for line in f:
            file_name, text1, text2 = line.strip().split('|')
            if os.path.exists(f'{teacher_path}/alignments/{file_name}.pkl'):
                filepaths_and_text.append( (file_name, text1, text2) )
    return filepaths_and_text


class TextMelSet(torch.utils.data.Dataset):
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text, hparams.teacher_dir)
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self.data_type=hparams.data_type

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        file_name = audiopath_and_text[0][:10]
        seq = os.path.join(hparams.data_path, self.data_type)
        mel = os.path.join(hparams.data_path, 'melspectrogram')
            
        with open(f'{seq}/{file_name}_sequence.pkl', 'rb') as f:
            text = pkl.load(f)
        
        if hparams.distillation==True:
            with open(f'{hparams.teacher_dir}/targets/{file_name}.pkl', 'rb') as f:
                mel = pkl.load(f)
        else:
            with open(f'{mel}/{file_name}_melspectrogram.pkl', 'rb') as f:
                mel = pkl.load(f)
                
        with open(f'{hparams.teacher_dir}/alignments/{file_name}.pkl', 'rb') as f:
            alignments = pkl.load(f)
        
        return (text, mel, alignments)

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    def __init__(self):
        return

    def __call__(self, batch):
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.zeros(len(batch), max_input_len, dtype=torch.long)
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])

        # include Spec padded and gate padded
        mel_padded = torch.zeros(len(batch), num_mels, max_target_len)
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)
            
        # include Spec padded and gate padded
        align_padded = torch.zeros(len(batch), max_target_len, max_input_len)
        for i in range(len(ids_sorted_decreasing)):
            align = batch[ids_sorted_decreasing[i]][2]
            align_padded[i, :align.size(0), :align.size(1)] = align
            
        return text_padded, input_lengths, mel_padded, output_lengths, align_padded

    
    
  
    


