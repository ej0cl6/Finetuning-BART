import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.models.bart.modeling_bart import BartEncoder, BaseModelOutput, _expand_mask
import ipdb

class FinetuningBART(nn.Module):
    def __init__(self, config, debug=True):
        super().__init__()
        self.config = config
        self.tokenizer = BartTokenizer.from_pretrained(self.config.pretrained_model, cache_dir=self.config.cache_dir)
        self.model = BartForConditionalGeneration.from_pretrained(self.config.pretrained_model, cache_dir=self.config.cache_dir)
        self.debug = debug
        if self.debug:
            self.show_demo_examples = True
    
    def save(self, path):
        self.model.save_pretrained(path)
        
    def process_data(self, src_sents, tgt_sents=None):
        # encoder inputs
        input_texts = src_sents
        inputs = self.tokenizer(src_sents, return_tensors='pt', truncation=True, padding=True)
        
        enc_idxs = inputs['input_ids']
        enc_attn = inputs['attention_mask']
        
        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        
        if tgt_sents is None:
            return enc_idxs, enc_attn, None, None, None
        
        # decoder inputs
        output_texts = tgt_sents
        outputs = self.tokenizer(tgt_sents, return_tensors='pt', padding=True)
        
        batch_size = enc_idxs.size(0)
        padding = torch.ones((batch_size, 1), dtype=torch.long)
        padding[:] = self.tokenizer.eos_token_id
        dec_idxs = torch.cat((padding, outputs['input_ids']), dim=1)
        dec_attn = torch.cat((torch.ones((batch_size, 1), dtype=torch.long), outputs['attention_mask']), dim=1)
            
        # labels
        padding = torch.ones((batch_size, 1), dtype=torch.long)
        padding[:] = self.tokenizer.pad_token_id
        raw_lbl_idxs = torch.cat((dec_idxs[:, 1:], padding), dim=1)
        lbl_attn = torch.cat((dec_attn[:, 1:], torch.zeros((batch_size, 1), dtype=torch.long)), dim=1)
        lbl_idxs = raw_lbl_idxs.masked_fill(lbl_attn==0, -100) # ignore padding
        
        dec_idxs = dec_idxs.cuda()
        dec_attn = dec_attn.cuda()
        lbl_idxs = lbl_idxs.cuda()
        
        if self.show_demo_examples:
            print()
            for i in range(3):
                print(f"IN:\n {input_texts[i]}")
                print(f"OUT:\n {output_texts[i]}")
            self.show_demo_examples = False
        
        return enc_idxs, enc_attn, dec_idxs, dec_attn, lbl_idxs
    
    def forward(self, src_sents, tgt_sents):
        enc_idxs, enc_attn, dec_idxs, dec_attn, lbl_idxs = self.process_data(src_sents, tgt_sents)
        
        outputs = self.model(input_ids=enc_idxs, 
                             attention_mask=enc_attn, 
                             decoder_input_ids=dec_idxs, 
                             decoder_attention_mask=dec_attn, 
                             labels=lbl_idxs, 
                             return_dict=True)
        
        loss = outputs['loss']
        
        return loss
    
    def generate(self, src_sents, num_beams=4):
        
        self.eval()
        
        max_length = self.config.max_tgt_len
        
        enc_idxs, enc_attn, _, _, _ = self.process_data(src_sents)
        with torch.no_grad():
            outputs = self.model.generate(input_ids=enc_idxs, 
                                          attention_mask=enc_attn, 
                                          num_beams=num_beams, 
                                          max_length=max_length)
        
        final_outputs = []
        for output in outputs:
            final_output = self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            final_outputs.append(final_output.strip())
            
        self.train()
        
        return final_outputs

