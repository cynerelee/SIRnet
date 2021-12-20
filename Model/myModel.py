import cv2
import torch
from torch import nn, Tensor
import torchvision.transforms as transforms#转换图片
#from Model.feat_extractor_backbone import build_backbone
#from Model.feat_extractor_tokenizer import build_tokenizer
from Model.set_decoder import SetDecoder
from Model.set_criterion import SetCriterion
from Model.my_mlp import MLPMixer#
import os
from transformers.modeling_bert import BertModel

class Mymodel(nn.Module):
    """
    Transformer computes self (intra image) and cross (inter image) attention
    """
    def __init__(self, ngt, num_classes, hidden_size, loss_weight):
        super().__init__()
        self.ngt = ngt
        self.num_classes = num_classes
        self.backbone = MLPMixer(
            image_size_H = 256,
            image_size_W = 256,
            channels = 3,
            patch_size = 16,
            dim = 512,
            depth = 12,
        )
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.bert.config.hidden_size=128
        self.bert.config.num_attention_heads=8
        self.config = self.bert.config
        
        self.fc = nn.Linear(512, self.config.hidden_size)
        #self.tokenizer = build_tokenizer([4, 4, 4, 4], [64, 128, 128], 128, 4)
        self.decoder = SetDecoder(ngt, num_classes, self.config)
        self.criterion = SetCriterion(num_classes, loss_weight)

       


    def forward(self, x, triplet=None,instrumentList=None,verbList=None,targetList=None,instrumentGT=None,verbGT=None,targetGT=None,mask=None):
       
        bz = x.size()[0]
        feat = self.backbone(x)
        feat = self.fc(feat)   

        attention_mask = torch.ones((bz, feat.size()[1]), requires_grad=False, dtype=torch.float32)
        outputs = self.decoder(enc_hs=feat, attention_mask=attention_mask)
        

        if triplet is not None:
            loss = self.criterion(outputs, triplet,instrumentList,verbList,targetList,instrumentGT,verbGT,targetGT,mask)
            return loss, outputs
        else:
            return outputs
    @torch.no_grad()
    def gen_triples(self, x):
        bz = x.size()[0]
        feat = self.backbone(x)
        
        feat = self.fc(feat)   

        attention_mask = torch.ones((bz, feat.size()[1]), requires_grad=False, dtype=torch.float32)
        outputs = self.decoder(enc_hs=feat, attention_mask=attention_mask)
        return outputs


