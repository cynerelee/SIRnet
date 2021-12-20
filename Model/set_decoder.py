import torch.nn as nn
import torch
from transformers.modeling_bert import BertIntermediate, BertModel, BertOutput, BertAttention, BertLayerNorm, BertSelfAttention

from torch.nn import functional as F
import os
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class SetDecoder(nn.Module):
    def __init__(self, ngt, num_classes, config, return_intermediate=False):
        super(SetDecoder,self).__init__()

        # self.config.hidden_size = hidden_size
        # self.config.num_attention_heads = 12
        self.return_intermediate = return_intermediate
        self.ngt = ngt
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(3)])
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
        self.query_embed = nn.Embedding(ngt, config.hidden_size)
        self.decoder2head = nn.Linear(config.hidden_size, num_classes[0] )
        self.decoder2rel = nn.Linear(config.hidden_size, num_classes[1])
        self.decoder2tail = nn.Linear(config.hidden_size, num_classes[2])
        #self.head2I=nn.Linear(num_classes[0]*ngt, num_classes[0]-1)
        #self.rel2V=nn.Linear(num_classes[1]*ngt, num_classes[1])
        #self.tail2T=nn.Linear(num_classes[2]*ngt, num_classes[2])
        # self.I2Triplet=nn.Linear(num_classes[0], num_classes[3])
        # self.V2Triplet=nn.Linear(num_classes[1], num_classes[3])
        # self.T2Triplet=nn.Linear(num_classes[2], num_classes[3])
        self.Triplet=nn.Linear(num_classes[0]+num_classes[1]+num_classes[2], num_classes[3])
        self.mlp=nn.Sequential( 
            Flatten(),
            nn.Linear(self.ngt, 3),
            nn.ReLU(),
            nn.Linear(3, self.ngt)            
            )
        
        torch.nn.init.orthogonal_(self.query_embed.weight, gain=1)
       
        self.weightIVT=nn.Parameter(torch.FloatTensor([1,1,1]))
        

    def forward(self, enc_hs, attention_mask):
        bz = enc_hs.size()[0]
        hidden_states = self.query_embed.weight.unsqueeze(0).repeat(bz, 1, 1)
        hidden_states = self.dropout(self.LayerNorm(hidden_states))
        all_hidden_states = ()

        for i, layer_module in enumerate(self.layers):
            if self.return_intermediate:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states, enc_hs, attention_mask
            )
            hidden_states = layer_outputs[0]

        pred_head = self.decoder2head(hidden_states)
        
        pred_rel = self.decoder2rel(hidden_states)
        pred_tail = self.decoder2tail(hidden_states)

        weight = F.avg_pool1d(pred_head,kernel_size=pred_head.size(2))#12,4,1
        weight+=F.avg_pool1d(pred_rel,kernel_size=pred_rel.size(2))#12,4,1
        weight+=F.avg_pool1d(pred_tail,kernel_size=pred_tail.size(2))#12,4,1
    
        weight+= F.max_pool1d(pred_head,kernel_size=pred_head.size(2))#12,4,1
        weight+=F.max_pool1d(pred_rel,kernel_size=pred_rel.size(2))#12,4,1
        weight+=F.max_pool1d(pred_tail,kernel_size=pred_tail.size(2))#12,4,1       
        weight=weight.view(bz,-1)
        weight=self.mlp(weight)#bz,ngt
        pred_head_matrix=pred_head.mul(weight.unsqueeze(2).repeat(1,1,pred_head.size(2)))
        pred_rel_matrix=pred_rel.mul(weight.unsqueeze(2).repeat(1,1,pred_rel.size(2)))
        pred_tail_matrix=pred_tail.mul(weight.unsqueeze(2).repeat(1,1,pred_tail.size(2)))#shape:[bz,ngt,class]
        pred_head_matrix=pred_head_matrix*self.weightIVT[0]
        pred_rel_matrix=pred_rel_matrix*self.weightIVT[1]
        pred_tail_matrix=pred_tail_matrix*self.weightIVT[2]
        matrix=torch.cat((pred_head_matrix,pred_rel_matrix,pred_tail_matrix),2)
       
        
        matrix=torch.mean(matrix,1)
        
        IVT=self.Triplet(matrix)

        
        return pred_head, pred_rel,pred_tail,IVT


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, enc_hs, attention_mask):
        
        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        encoder_batch_size, encoder_sequence_length, _ = enc_hs.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if attention_mask.dim() == 3:
            encoder_extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            encoder_extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    encoder_hidden_shape, attention_mask.shape
                )
            )
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        encoder_extended_attention_mask=encoder_extended_attention_mask.to(hidden_states.device)

        cross_attention_outputs = self.crossattention(
            hidden_states=attention_output, encoder_hidden_states=enc_hs,
            encoder_attention_mask=encoder_extended_attention_mask
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

