import torch.nn as nn
from models.Attention import MultiHeadedAttention
from models.MLP import MLP
from tools.utils import *
import torch.nn.functional as F



class ComAttention(nn.Module):

    def __init__(self, head, d_model):
        super(ComAttention, self).__init__()
        self.d_k = d_model // head
        self.head = head
        self.q_liner = nn.Linear(d_model, d_model)
        self.k_liner = nn.Linear(d_model, d_model)
        self.fusion = nn.Sequential(
            nn.Conv2d(head,2*head,1),
            nn.Conv2d(2*head,head,1),
            nn.Conv2d(head,1,1)
        )
        self.p_attn = MultiHeadedAttention(head, d_model)
        self.n_attn = MultiHeadedAttention(head, d_model)

        self.gate_n = nn.Linear(d_model, d_model)
        self.gate_p = nn.Linear(d_model, d_model)

        self.v_n = nn.Linear(d_model,d_model)
        self.v_p = nn.Linear(d_model,d_model)

    def forward(self, feature, data_mask, pm=None, nm=None):
        nbatches = feature.size(0)
        q = self.q_liner(feature).view(nbatches, -1, self.head, self.d_k).transpose(1, 2)
        k = self.k_liner(feature).view(nbatches, -1, self.head, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_k)
        scores = torch.sigmoid(self.fusion(scores))

        zero = torch.zeros_like(scores).cuda()
        one = torch.ones_like(scores).cuda()
        p_mask = torch.where(scores > 0.5,one,zero) * scores
        n_mask = torch.where(scores <= 0.5,one,zero) * scores
        p,pm = self.p_attn(feature, feature, feature, p_mask.squeeze(1) * data_mask, pm)
        n,nm = self.n_attn(feature, feature, feature, n_mask.squeeze(1) * data_mask, nm)

        vp = self.v_p(p)
        vn = self.v_n(n)
        ep = self.gate_p(p)
        en = self.gate_p(n)
        v = torch.stack((vp,vn),dim=0)
        e = F.softmax(torch.stack((ep,en),dim=0),dim=0)
        out_feature = torch.sum((v * e),dim=0)
        return out_feature,pm,nm



class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()
        self.embed = nn.Linear(2, 32)
        self.embed2 = nn.Linear(2, 32)
        self.TRmodel = ComAttention(1, 32)
        self.SRmodel = ComAttention(1, 32)
        
        self.TRmodel2 = ComAttention(4, 32)
        self.SRmodel2 = ComAttention(4, 32)

        self.add_spa = nn.Linear(258, 256)
        self.add_spa2 = nn.Linear(258, 256)
        self.gen_others = nn.Sequential(
            MLP(18, 64, [128, 256, 128]),
            nn.Linear(64, 22)
        )
        self.gen_multi_dest = nn.Sequential(
            MLP(256, 64, [128,256,128]),
            nn.Linear(64,100)
        )

    def past_encoder(self, abs_traj, initial_pos, scene_mask):

        peds = abs_traj.shape[0]
        temporal_feature = self.embed(abs_traj[:, :, :] - abs_traj[:, -1:, :])
        spatio_feature = self.embed2(abs_traj)
        feature = spatio_feature + temporal_feature
        pm = torch.zeros(1,1,peds,peds).cuda()
        nm = torch.zeros(1,1,peds,peds).cuda()
        final_feature = torch.zeros_like(feature).cuda()
        tmp_feature = torch.zeros_like(feature).cuda()
        temporal_mask = [subsequent_mask(peds, i).cuda() for i in range(1,9)]
        mask = []
        for t in range(8):
            tmp,pm,nm = self.SRmodel(feature[:, t:t+1, :].permute(1,0,2),scene_mask.repeat(1,1,1),pm,nm)
            mask.append(torch.stack([pm.clone().squeeze(),nm.clone().squeeze()],dim=0))
            final_feature[:, :t+1, :],_,_ = self.TRmodel(torch.cat((final_feature[:,:t,:],tmp.permute(1,0,2)),dim=1),temporal_mask[t])
        mask = torch.stack(mask,dim=0)
        return final_feature + temporal_feature, [abs_traj,mask]
        # return temporal_feature
    
    def forward(self, traj, initial_pos, scene_mask, dest_gt, train = True):
        
        peds = traj.size(0)
        past_feature,visdata = self.past_encoder(traj, initial_pos, scene_mask)
        past_feature = past_feature.contiguous().view(peds, -1)
        multi_dest = self.gen_multi_dest(past_feature).view(-1,20,5)
        if train:
            other_input = torch.cat((traj-traj[:, -1:, :], dest_gt),dim=1).contiguous().view(-1,18)
            others = self.gen_others(other_input).view(-1,11,2)
            return multi_dest, others
        else:
            return multi_dest,visdata
        # return past_feature
    
    def predict(self, dest, traj):
        other_input = torch.cat((traj, dest),dim=1).contiguous().view(-1,18)
        others = self.gen_others(other_input).view(-1,11,2)
        return others