import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
 
class MultipleLogitAdjustment(nn.Module):
    def __init__(self, cls_num_list=None,  max_m=0.5, s=30, tau=2):
        super().__init__()
        self.base_loss = F.cross_entropy 
     
        prior = np.array(cls_num_list) / np.sum(cls_num_list)
        self.prior = torch.tensor(prior).float().cuda()
        self.C_number = len(cls_num_list)  # class number
        self.s = s
        self.tau = tau 

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)  # output_logits indicates the final prediction

        loss = 0

        # Obtain logits from each expert  
        expert1_logits = extra_info['logits'][0]
        expert2_logits = extra_info['logits'][1] 
        expert3_logits = extra_info['logits'][2]  
 
        # Softmax loss for expert 1 
        loss += self.base_loss(expert1_logits, target)
        
        # Balanced Softmax loss for expert 2 
        expert2_logits = expert2_logits + torch.log(self.prior + 1e-9) 
        loss += self.base_loss(expert2_logits, target)
        
        # Softmax loss for expert 3
        expert3_logits = expert3_logits + 2.6*torch.log(self.prior + 1e-9) 
        loss += self.base_loss(expert3_logits, target)
   
        return loss
    
 
     