import os
import math
from numpy.core.fromnumeric import var
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import copy
import networkx as nx
import warnings
import torch.distributions.multivariate_normal as torchdist
warnings.filterwarnings("ignore", category=Warning)


def kmeans(x, ncluster, niter=10):
    '''
    x : torch.tensor(data_num,data_dim)
    ncluster : The number of clustering for data_num
    niter : Number of iterations for kmeans
    '''
    N, D = x.size()
    c = x[torch.randperm(N)[:ncluster]] # init clusters at random
    for i in range(niter):
        # assign all pixels to the closest codebook element
        # .argmin(1) : 按列取最小值的下标,下面这行的意思是将x.size(0)个数据点归类到random选出的ncluster类
        a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1)
        # move each codebook element to be the mean of the pixels that assigned to it
        # 计算每一类的迭代中心，然后重新把第一轮随机选出的聚类中心移到这一类的中心处
        c = torch.stack([x[a==k].mean(0) for k in range(ncluster)])
        # re-assign any poorly positioned codebook elements
        nanix = torch.any(torch.isnan(c), dim=1)
        ndead = nanix.sum().item()
        # print('done step %d/%d, re-initialized %d dead clusters' % (i+1, niter, ndead))
        c[nanix] = x[torch.randperm(N)[:ndead]] # re-init dead clusters
    return c


def norm(traj,initial_pos,scale):
    minx = torch.min(initial_pos[:, 0])
    miny = torch.min(initial_pos[:, 1])
    initial_pos[:, 0] = initial_pos[:, 0] - minx
    initial_pos[:, 1] = initial_pos[:, 1] - miny
    traj = traj * scale
    initial_pos = initial_pos * scale
    
    return initial_pos, traj

def bivariate_loss(V_pred,V_trgt):
    #mux, muy, sx, sy, corr
    #assert V_pred.shape == V_trgt.shape
    normx = V_trgt[:,:,0]- V_pred[:,:,0]
    normy = V_trgt[:,:,1]- V_pred[:,:,1]
    epsilon = 1e-20

    sx = torch.exp(V_pred[:,:,2]) #sx
    sy = torch.exp(V_pred[:,:,3]) #sy
    corr = torch.tanh(V_pred[:,:,4]) #corr
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability

    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)
    
    return result


def gmm_loss(V_pred, V_pred_gt, indi=[1]):

    loss = torch.zeros((len(indi))).cuda()
    for i in range(len(indi)):
        loss[i] = bivariate_loss(V_pred[:, i:i+1, :], V_pred_gt.clone())
    indicate = torch.tensor(indi).cuda()
    result = torch.sum(loss * indicate)

    return result

def dlow_loss(gt, dest):
    gt = gt.repeat(1,20,1)
    loss = (gt - dest)**2
    loss = torch.sum(loss,dim=-1)
    loss = torch.sqrt(loss)
    _,mindex = torch.min(loss,1)
    final_loss = torch.zeros((loss.size(0)))
    for i in range(loss.size(0)):
        final_loss[i] = loss[i, mindex[i]]
    return torch.mean(final_loss)

def one_of_many_loss(V_pred, V_pred_gt):
    loss = torch.zeros(20).cuda()
    for i in range(20):
        loss[i] = bivariate_loss(V_pred[:, i:i+1, :], V_pred_gt.clone())
    _,mindex = torch.min(loss,0)
    return loss[mindex]


def fde_all(pred, gt):
    '''
    pred: samples * peds * length * features
    gt: same as pred
    '''
    truth_20, pre_20 = gt.permute(1,2,0,3), pred.permute(1,2,0,3)
    err_20 = truth_20 - pre_20
    err_20 = err_20**2
    err_20 = torch.sqrt(err_20.sum(dim=3))

    err_fde = err_20[:, -1, :]
    fde_cnt = torch.min(err_fde, dim=1)[1]
    err_fde = torch.min(err_fde, dim=1)[0].sum()

    return err_fde, fde_cnt

def ade_all(pred, gt, length=12):
    '''
    pred: samples * peds * length * features
    gt: same as pred
    '''
    truth_20, pre_20 = gt.permute(1,2,0,3), pred.permute(1,2,0,3)
    err_20 = truth_20 - pre_20
    err_20 = err_20**2
    err_20 = torch.sqrt(err_20.sum(dim=3))


    err_ade = err_20.sum(dim=1) / length
    ade_cnt = torch.min(err_ade, dim=1)[1]
    err_ade = torch.min(err_ade, dim=1)[0].sum()

    return err_ade, ade_cnt


def get_gauss_dis(traj):
    '''
    traj: peds * length * gauss_parameters(5)
    '''
    sx = torch.exp(traj[:,:,2]) #sx
    sy = torch.exp(traj[:,:,3]) #sy
    corr = torch.tanh(traj[:,:,4]) #corr
    
    cov = torch.zeros(traj.shape[0],traj.shape[1],2,2).cuda()
    cov[:,:,0,0]= sx*sx
    cov[:,:,0,1]= corr*sx*sy
    cov[:,:,1,0]= corr*sx*sy
    cov[:,:,1,1]= sy*sy
    mean = traj[:,:,0:2]

    mvnormal = torchdist.MultivariateNormal(mean,cov)

    return mvnormal


def subsequent_mask(batch, size):
    "Mask out subsequent positions."
    attn_shape = (batch, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class DataSets(Dataset):
    '''
    Dataloader for the trajectory datasets
    Each Dataloader for one trajectory datasets such as eth, hotel, univ, zra1, zara2
    '''
    def __init__(self, data_raw_dir, obs_len=8, pred_len=12, need_frame=False):
        '''
        Arguments:
        - data_raw_dir: Directory of each pedestrian's raw trajectory datasets
            data format: <frame_id> <ped_id> <x> <y>
        - obs_len: Number of frames in input trajectories
        - pred_len: Number of frames in output trajectories
        '''
        super(DataSets, self).__init__()

        self.data_raw_dir = data_raw_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.abs_obs_data = []
        self.abs_pre_data = []
        self.rel_obs_data = []
        self.rel_pre_data = []
        self.shift_data = []
        self.start_data = []
        self.final_mask = []
        self.min_x = []
        self.min_y = []
        #load all data directory in datasets
        all_datasets_dir = os.listdir(self.data_raw_dir)
        all_datasets_dir = [os.path.join(self.data_raw_dir, path) for path in all_datasets_dir]

        #Processing every file in dataset's {train/val/test} data
        for each_datafile_path in all_datasets_dir:
            print('Processing file: '+each_datafile_path)
            raw_data = self.load_data_from_file(each_datafile_path)
            #get all numbers of frames
            frames = np.unique(raw_data[:, 0]).tolist()
            #add a top dimension order by frame
            data_orderbyframes = []
            for frame in frames:
                data_orderbyframes.append(raw_data[frame == raw_data[: ,0], :])
            #get the number of sequences with each sequence include 20 frames
            num_sequences = int(math.ceil(len(frames) - self.seq_len))
            #Processing every sequence data
            for seq_id in tqdm(range(0, num_sequences + 1)):
                final_seq_data = []
                #concatenate 20 frames data in a array
                curr_seq_data = np.concatenate(data_orderbyframes[seq_id:seq_id + self.seq_len], axis=0)
                #get all pedestrian's id
                peds_id_in_curr_seq = np.unique(curr_seq_data[:, 1])
                #Processing every person data in every sequence
                for _, ped_id in enumerate(peds_id_in_curr_seq):
                    curr_ped_data = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    #if person data frames < 20, not use it
                    if len(curr_ped_data) != self.seq_len:
                        continue
                    curr_ped_data = np.concatenate([curr_ped_data[:, :1], curr_ped_data[:, 2:]], axis=1)
                    final_seq_data.append(curr_ped_data)
                #if there are no person in a sequence, not use it
                if len(final_seq_data) <= 1:
                    continue
                abs_seq_data = np.asarray(final_seq_data)
                self.min_x.append(copy.deepcopy(np.min(abs_seq_data[:, :, 1])).astype(np.float32))
                self.min_y.append(copy.deepcopy(np.min(abs_seq_data[:, :, 2])).astype(np.float32))
                abs_seq_data[:, :, 1] = abs_seq_data[:, :, 1] - np.min(abs_seq_data[:, :, 1])
                abs_seq_data[:, :, 2] = abs_seq_data[:, :, 2] - np.min(abs_seq_data[:, :, 2])
                rel_seq_data, shift_seq_data, start_seq_data = self.get_rel_shift(abs_seq_data)
                # mask = self.get_social_mask(abs_seq_data)
                if not need_frame:
                    abs_seq_data = abs_seq_data[:, :, 1:]
                    rel_seq_data = rel_seq_data[:, :, 1:]
                    shift_seq_data = shift_seq_data[:, :, 1:]
                    start_seq_data = start_seq_data[:, :, 1:]
                self.rel_obs_data.append(copy.deepcopy(rel_seq_data[:, :self.obs_len, :]).astype(np.float32))
                self.rel_pre_data.append(copy.deepcopy(rel_seq_data[:, self.obs_len:, :]).astype(np.float32))
                self.shift_data.append(copy.deepcopy(shift_seq_data).astype(np.float32))
                self.start_data.append(copy.deepcopy(start_seq_data).astype(np.float32))
                self.abs_obs_data.append(copy.deepcopy(abs_seq_data[:, :self.obs_len, :]).astype(np.float32))
                self.abs_pre_data.append(copy.deepcopy(abs_seq_data[:, self.obs_len:, :]).astype(np.float32))
                # self.final_mask.append(copy.deepcopy(mask).astype(np.float32))
        self.abs_obs_data = np.asarray(self.abs_obs_data)
        self.abs_pre_data = np.asarray(self.abs_pre_data)
        self.rel_obs_data = np.asarray(self.rel_obs_data)
        self.rel_pre_data = np.asarray(self.rel_pre_data)
        self.min_x = np.asarray(self.min_x)
        self.min_y = np.asarray(self.min_y)
        self.norm_obs_data = self.abs_obs_data - self.shift_data
        self.norm_pre_data = self.abs_pre_data - self.shift_data

    def get_rel_shift(self, abs_seq_data):
        _rel = copy.deepcopy(abs_seq_data)
        _rel[:, 1:, 1:] = abs_seq_data[:, :-1, 1:]
        start = abs_seq_data[:, :1, :]
        shift = abs_seq_data[:, self.obs_len-1:self.obs_len, :]
        rel = copy.deepcopy(abs_seq_data)
        rel[:, :, 1:] = abs_seq_data[:, :, 1:] - _rel[:, :, 1:]
        rel = np.asarray(rel)
        shift = np.asarray(shift)
        return rel, shift, start

    def load_data_from_file(self,data_path):
        data = []
        with open(data_path, 'r') as r:
            for line in r:
                line = line.strip().split('\t')
                line = [float(i) for i in line]
                data.append(line)
        return np.asarray(data)

    def get_social_mask(self, data):
        peds = data.shape[0]
        l2_mask = np.zeros((self.obs_len, peds,peds))
        l2_mask[0, :, :] = (np.ones((peds,peds)) - np.eye(peds))
        cos_mask = np.zeros((self.obs_len, peds,peds))
        cos_mask[0, :, :] = (np.ones((peds,peds)) - np.eye(peds))
        traj = data[:, :self.obs_len, 1:]
        norm_traj = traj - traj[:, :1, :]
        for f in range(1,self.obs_len):
            for h in range(peds):
                for k in range(h+1,peds):
                    l2_w = self.get_l2_w(traj[h, f, :], traj[k, f, :])
                    cos_w = self.get_cos_w(norm_traj[h, :f, :], norm_traj[k, :f, :])
                    l2_mask[f,h,k] = l2_w
                    cos_mask[f,h,k] = cos_w
        mask = np.stack((cos_mask,l2_mask),axis=3)
        for f in range(1,self.obs_len):
            cos_G = nx.from_numpy_matrix(mask[f, :, :, 0])
            mask[f, :, :, 0] = nx.to_numpy_matrix(cos_G).A
            l2_G = nx.from_numpy_matrix(mask[f, :, :, 1])
            mask[f, :, :, 1] = nx.to_numpy_matrix(l2_G).A
        return mask

    
    def get_l2_w(self, t1, t2):
        vector_a = t1.reshape(-1)
        vector_b = t2.reshape(-1)
        norm = math.sqrt(sum((vector_a-vector_b)**2))
        if norm == 0:
            return 0
        return 1/norm

    def get_cos_w(self, t1,t2):
        vector_a = t1.reshape(-1)
        vector_b = t2.reshape(-1)
        vector_a = np.mat(vector_a)
        vector_b = np.mat(vector_b)
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        if denom == 0:
            cos = 1
        else:
            cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim



    def __len__(self):
        return len(self.abs_obs_data)

    def __getitem__(self, index):
        out = [
            self.abs_obs_data[index], self.abs_pre_data[index], \
            self.rel_obs_data[index], self.rel_pre_data[index], \
            self.norm_obs_data[index], self.norm_pre_data[index], \
            self.shift_data[index], self.start_data[index],\
            self.min_x[index], self.min_y[index]
            # self.final_mask[index]
        ]
        return out