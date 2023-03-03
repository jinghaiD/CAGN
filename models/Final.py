import re
from grpc import Status
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
        self.p_attn = MultiHeadedAttention(4, 32)
        self.n_attn = MultiHeadedAttention(4, 32)

        self.gate_n = nn.Linear(d_model, d_model)
        self.gate_p = nn.Linear(d_model, d_model)

        self.v_n = nn.Linear(d_model,d_model)
        self.v_p = nn.Linear(d_model,d_model)

    def forward(self, feature, data_mask, pm=None, nm=None):
        nbatches = feature.size(0)
        q = self.q_liner(feature).view(nbatches, -1, self.head, self.d_k).transpose(1, 2)
        k = self.k_liner(feature).view(nbatches, -1, self.head, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_k)
        test_sco = torch.softmax(self.fusion(scores),dim=-1)
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

        return out_feature
        # return p

class DisGate(nn.Module):

    def __init__(self, d_model):
        super(DisGate, self).__init__()
        
        self.attn = MultiHeadedAttention(4, 32)

        self.gate_dis = nn.Linear(d_model, d_model)
        self.gate_ans = nn.Linear(d_model, d_model)

        self.v_dis = nn.Linear(d_model,d_model)
        self.v_ans = nn.Linear(d_model,d_model)


    def forward(self, dis, ans):

        peds = dis.size(0)
        _one = torch.zeros((peds,1,32)).cuda()
        ans = torch.cat((ans,_one),dim=1)
        ans,_ = self.attn(ans,ans,ans)
        ans = ans[:, -1:, :]
        vp = self.v_dis(dis)
        vn = self.v_ans(ans)
        ep = self.gate_dis(dis)
        en = self.gate_ans(ans)
        v = torch.stack((vp,vn),dim=0)
        e = F.softmax(torch.stack((ep,en),dim=0),dim=0)
        out_feature = torch.sum((v * e),dim=0)

        return out_feature
        # return p



class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()
        self.embed = nn.Linear(2, 32)
        self.embed2 = nn.Linear(2, 32)
        self.TRmodel = ComAttention(4, 32)
        self.SRmodel = ComAttention(4, 32)
        
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
        self.dis_gate = DisGate(32)

    def past_encoder(self, norm_traj, initial_pos, scene_mask):

        # peds = norm_traj.shape[0]
        # norm_traj = self.embed2(norm_traj)
        # init_traj = norm_traj - norm_traj[:, 0:1, :]
        # v = torch.norm(init_traj[:, 1, :],dim=1)
        # a = torch.norm((init_traj - init_traj[:, 1:2, :])[:, 1, :]) - v
        # init_status = torch.stack((v,a),dim=1)
        # re_status = self.embed(init_status).unsqueeze(1)
        # temporal_mask = [subsequent_mask(peds, i).cuda() for i in range(2,10)]
        # ps = torch.zeros((1,4,peds,peds)).cuda()
        # ns = torch.zeros((1,4,peds,peds)).cuda()

        # # dis = torch.randn((peds,1,32)).cuda()

        # for i in range(norm_traj.shape[1]):
        #     re_status = torch.cat((re_status, norm_traj[:, i:i+1, :]),dim=1)
        #     temporal_feature,pt,nt = self.TRmodel(re_status, temporal_mask[i])
        #     temporal_feature = temporal_feature + re_status
        #     abs_pos = self.add_spa(torch.cat((temporal_feature[:, -1, :], abs_traj[:, i, :]),dim=1)).unsqueeze(1)
        #     feature, ps, ns = self.SRmodel(abs_pos.permute(1,0,2), scene_mask.unsqueeze(0), ps, ns)
        #     feature = feature.permute(1,0,2) + abs_pos
        #     re_status = torch.cat((temporal_feature[:, :-1, :], feature),dim=1)
        #     # dis = self.dis_gate(dis, re_status)


        obs_traj = self.embed(norm_traj)
        # with torch.no_grad():
        peds = norm_traj.shape[0]
        temporal_mask = subsequent_mask(peds, 8).cuda()

        #Ours model
        temporal_feature = self.TRmodel(obs_traj, temporal_mask)
        temporal_feature = temporal_feature + obs_traj
        temporal_feature = temporal_feature.view(peds, -1)
        temporal_feature = self.add_spa(torch.cat((temporal_feature, initial_pos),dim=1)).view(peds, 8, 32).permute(1,0,2)
        feature = self.SRmodel(temporal_feature, scene_mask.repeat(8,1,1))
        feature = feature.permute(1,0,2) + obs_traj


        temporal_feature = self.TRmodel2(feature, temporal_mask)
        temporal_feature = temporal_feature + obs_traj
        temporal_feature = temporal_feature.view(peds, -1)
        temporal_feature = self.add_spa2(torch.cat((temporal_feature, initial_pos),dim=1)).view(peds, 8, 32).permute(1,0,2)
        feature = self.SRmodel2(temporal_feature, scene_mask.repeat(8,1,1))
        feature = feature.permute(1,0,2) + obs_traj

        return feature
        # return re_status
        # return dis
    
    def forward(self, traj, initial_pos, scene_mask, dest_gt, train = True):
        
        peds = traj.size(0)
        past_feature = self.past_encoder(traj, initial_pos, scene_mask)
        past_feature = past_feature.contiguous().view(peds, -1)
        multi_dest = self.gen_multi_dest(past_feature).view(-1,20,5)
        if train:
            other_input = torch.cat((traj, dest_gt),dim=1).contiguous().view(-1,18)
            others = self.gen_others(other_input).view(-1,11,2)
            return multi_dest, others
        else:
            return multi_dest
    
    def predict(self, dest, traj):
        other_input = torch.cat((traj, dest),dim=1).contiguous().view(-1,18)
        others = self.gen_others(other_input).view(-1,11,2)
        return others
#          ETH        HOTEL       UNIV         ZARA1       ZARA2
#K=5    0.65/1.26 | 0.22/0.43 | 0.47/0.92 | 0.33/0.70 | 0.27/0.58 0.39/0.78
#K=10 