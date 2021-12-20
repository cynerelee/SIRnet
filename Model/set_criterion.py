import torch.nn.functional as F
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']='0'
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def returnWeight(txtdir):
    instrumentcount=np.loadtxt(txtdir)
    instrumentweight = instrumentcount/np.sum(instrumentcount)
    instrumentweight = 1/np.log(1.1 + instrumentweight)
    instrumentweight = instrumentweight/instrumentweight.mean()
    return instrumentweight
class SetCriterion(nn.Module):


    def __init__(self, num_classes, loss_weight=[1,1,1],Total_loss_weight=[0.1,1]):

        super().__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.Total_loss_weight=Total_loss_weight
       

    def forward(self, outputs, triplet,instrumentId,verbId,targetId,instrument,verb,target,mask): 
        pred_head, pred_rel,pred_tail,IVT=outputs       
        pred_head= pred_head.unsqueeze(2)
        pred_rel=pred_rel.unsqueeze(2)
        pred_tail=pred_tail.unsqueeze(2)
        head_target = instrumentId.unsqueeze(2)
        rel_target = verbId.unsqueeze(2)
        tail_target = targetId.unsqueeze(2)
        instrumentweight=torch.tensor([1.32863889, 0.55159365, 1.27372328, 0.75904817, 1.3103314 ,
       1.49935446, 0.61570293, 1.29414463, 0.3674626],requires_grad=False)
        verbweight=torch.tensor([1.3454168,  1.17237702, 1.3454168,  1.17557095 ,1.13543288, 1.08983935, 0.55163586,
         0.35174053, 1.01637501, 1.37591964 ,0.56370784 ,1.23637941, 0.64018793],requires_grad=False)
        targetweight=torch.tensor([0.97457543, 1.02542457],requires_grad=False)
        criterion = nn.CrossEntropyLoss(weight=instrumentweight)
        criterion =criterion.to(device)
        criterion1 = nn.CrossEntropyLoss(weight=verbweight)
        criterion1 =criterion1.to(device)
        criterion2 = nn.CrossEntropyLoss(weight=targetweight)
        criterion2 =criterion2.to(device)
        bz, ngt = pred_head.size()[:2]
        cur_ngt = head_target.size()[1]
        sim_mat = torch.zeros((bz, ngt, cur_ngt), requires_grad=False, dtype=torch.float32)
        sim_mat=sim_mat.to(device)
        for i in range(bz):
            for j in range(ngt):
                for k in range(cur_ngt):
                    head_loss=criterion(pred_head[i, j], head_target[i, k].type(torch.long))
                    rel_loss =criterion1(pred_rel[i, j], rel_target[i, k].type(torch.long))
                    tail_loss =criterion2(pred_tail[i, j], tail_target[i, k].type(torch.long))                    
                    sim_mat[i, j, k] = self.loss_weight[0] * head_loss + self.loss_weight[1] * rel_loss + self.loss_weight[2] *tail_loss
        mask = mask.unsqueeze(1).repeat(1, ngt, 1)
        

        data1=torch.tensor((1-mask).data,dtype=torch.bool).to(device)
        sim_mat.data.masked_fill_(data1, float('inf'))
        sim_mat_cpu = sim_mat.cpu().detach().numpy()
        

        indices=[]
        for i, (c, k) in enumerate(zip(sim_mat_cpu, mask)):
            mat = c[:, :int(sum(k[0, :]))]  
            indices.append(linear_sum_assignment(mat))

       

        loss1 = 0
        count = 0
        for i in range(bz):
            for j in range(len(indices[i][0])):
                loss1 += sim_mat[i, indices[i][0][j], indices[i][1][j]]
                count += 1
        loss1=loss1/count
        tripletweight=torch.tensor([1.17236407 ,1.0166286,  1.19620973, 0.5544405 , 0.63531401, 0.51258428,
 1.08866652, 1.15795989, 1.07389395, 0.98728399, 1.12754142, 1.05953744,
 1.16945323 ,1.15512349, 1.02097204 ,1.15795989, 1.07147279, 0.50627649,
 1.07147279 ,0.61697221 ,1.16367678, 1.0231585 , 1.18416106, 1.04329092,
 1.10645159, 1.18416106 ,1.15795989, 1.16367678 ,0.73949534, 0.78760821,
 1.08617476 ,1.00805777],requires_grad=False)



        criterion4 = nn.BCEWithLogitsLoss(weight=tripletweight)
        criterion4 =criterion4.to(device)
        
        loss5=criterion4(IVT,triplet.type(torch.float32))
       
        loss=self.Total_loss_weight[0]*loss1+self.Total_loss_weight[1]*loss5
        return loss
