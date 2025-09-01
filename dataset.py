import torch
import torch.nn as nn
from torch.utils.data import Dataset

# dataset loader including inint, len, getitem
class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # dataset is a list of pairs of sentences
        # Each pair has a sentence in the source language and a sentence in the target language
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens
        # in one pass, tokenizer encoder split the text into tokens
        # and convert the tokens into ids (numbers in the vocabulary)
        # ".ids" returns only the ids (numbers in the vocabulary)
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        # We add [PAD] tokens to the end of 
        # the sentence untill it reaches the seq_len

        # For encoder side, we add both [SOS] and [EOS]
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        # But for decoder side, we only add [SOS], [EOS] is added to the label
        # Encoder is trained to produce contextualized representation, so encoder
        # needs to see the full sentence including [EOS],
        # but decoder is trained to predict the next token (auto-regressive)
        # that's why we don't add [EOS] to the decoder input
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add [SOS] or <s> and [EOS] or </s> token to encoder
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only [SOS] or <s> token to decoder
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only [EOS] </s> token to the label
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        # This is different from the "max seq len" hyperparameter
        # max seq_len is determined by the model and deppends
        # on limitations for positional embeddings computation
        # But in training, evaluation, and inference we need all tensors
        # to be of the same size for efficient prallelization on GPU   
        # The code usually defines a constant like seq_len, 
        # which is often ≤ max_seq_len of the architecture.    
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            # we don't want the model to attend to the padding tokens
            # any non-padding token return True (1), padding tokens return False (0)
            # unsqueeze to add batch and head dimension
            # At the end, the shape of encoder_mask is (1, 1, seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            # again, we don't want the decoder to attend to the padding tokens
            # also, we don't want the decoder to attend to the future tokens, causal mask does that
            # in the github code, it is mentioned like the following, one unsqueeze(0)  and (1, seq_len)
            # At the end, the shape of decoder_mask is (1, seq_len, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
def causal_mask(size):
    # torch.triu returns a upper triangular matrix
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    # we want the lower triangular matrix for causal mask
    # so we return True (1) where mask is 0
    return mask == 0