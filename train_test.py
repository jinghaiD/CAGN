from social_eth_ucy_utils import *
from models.Final import Model
from tools.utils import *
import argparse
from torch.utils.data.dataloader import DataLoader


parser = argparse.ArgumentParser(description='AAAI')
parser.add_argument('--set', default='hotel', type=str)
parser.add_argument('--gpu', default='4', type=str)

args = parser.parse_args()
sets_name = args.set
gpu = args.gpu
obs_len = 8
pre_len = 12
samples = 20
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
model = Model()
model.cuda()
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
l2loss = nn.MSELoss()
scale = 1
if sets_name == 'sdd':
    scale = 0.01
train_dataset = SocialDatasetETHUCY(set_name=sets_name, set_type='train', b_size=512, t_tresh=0, d_tresh=50)
test_dataset = SocialDatasetETHUCY(set_name=sets_name, set_type='test', b_size=512, t_tresh=0, d_tresh=50)
best_ans = 100000
best_epoch = 0
indicate = [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05]
# indicate = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
# indicate = [0.2,0.2,0.2,0.2,0.2]
# model_path = sets_name+'_best.pth'
# checkpoint = torch.load(model_path)
# model.load_state_dict(checkpoint)
for epoch in range(650):
    print('*'*60)
    print('Total epoch numbers: 650')
    print('current epoch numbers: '+str(epoch))
    cnt = 0
    epoch_loss = 0
    for i, (traj, mask, initial_pos) in enumerate(zip(train_dataset.trajectory_batches, train_dataset.mask_batches, train_dataset.initial_pos_batches)):
        traj, mask, initial_pos = torch.FloatTensor(traj).cuda(), torch.FloatTensor(mask).cuda(), torch.FloatTensor(initial_pos).cuda()
        initial_pos,traj = norm(traj,initial_pos,scale)
        norm_traj = traj - traj[:, obs_len-1:obs_len, :]
        norm_traj_obs = norm_traj[:, :obs_len, :]
        abs_traj_obs = traj[:, :obs_len, :]
        norm_traj_pre = norm_traj[:, obs_len:, :]
        norm_dest = norm_traj[:, -1:, :]
        dest_dis,others = model(norm_traj_obs, initial_pos, mask, norm_dest)
        loss = gmm_loss(dest_dis, norm_dest, indicate) + l2loss(others, norm_traj_pre[:, :-1, :]) + 4*one_of_many_loss(dest_dis, norm_dest)
        # loss = one_of_many_loss(dest_dis, norm_dest) + l2loss(others, norm_traj_pre[:, :-1, :])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cnt += 1
        epoch_loss += loss.item()
    epoch_loss = epoch_loss / cnt
    print('epoch loss(dest loss + l2 loss):'+str(epoch_loss))
    if epoch > -1:
        import time
        with torch.no_grad():
            all_peds = 0
            best_ade = 0
            best_fde = 0
            length = pre_len
            start = time.time()
            for i, (traj, mask, initial_pos) in enumerate(zip(test_dataset.trajectory_batches, test_dataset.mask_batches, test_dataset.initial_pos_batches)):
                traj, mask, initial_pos = torch.FloatTensor(traj).cuda(), torch.FloatTensor(mask).cuda(), torch.FloatTensor(initial_pos).cuda()
                initial_pos,traj = norm(traj,initial_pos,scale) 
                norm_traj = traj - traj[:, obs_len-1:obs_len, :]
                norm_traj_obs = norm_traj[:, :obs_len, :]
                abs_traj_obs = traj[:, :obs_len, :]
                norm_traj_pre = norm_traj[:, obs_len:, :]
                norm_dest = norm_traj[:, -1:, :]
                peds = traj.size(0)

                truth = torch.empty([samples, peds, length, 2]).cuda()
                for s in range(samples):
                    truth[s] = norm_traj_pre.clone()

                ans = torch.empty([samples, peds, length, 2]).cuda()
                multi_dest = model(norm_traj_obs, initial_pos, mask, norm_dest, train=False)
                for s in range(len(indicate)):
                    dis = get_gauss_dis(multi_dest[:, s:s+1, :])

                    for n in range(int(samples*indicate[s])):

                        sam_dest = dis.sample()
                        others = model.predict(sam_dest, norm_traj_obs)
                        others = torch.cat((others, sam_dest),dim=1)
                        ans[int(samples*sum(indicate[:s]))+n] = others.clone()

                err_ade, ade_cnt = ade_all(ans, truth, length)
                err_fde, fde_cnt = fde_all(ans, truth)

                all_peds = all_peds + peds
                best_ade = best_ade + err_ade
                best_fde = best_fde + err_fde
            end = time.time()
            # print(end-start)
            # exit()
            best_ade = best_ade / all_peds
            best_fde = best_fde / all_peds

            print('current epoch ADE:{:.6f}, FDE:{:.6f}'.format(best_ade / scale, best_fde / scale))
            if best_fde < best_ans:
                best_ans = best_fde
                ade_best = best_ade
                best_epoch = epoch
                # torch.save(model.state_dict(), sets_name+'_best.pth')
            print('The best ADE:{:.6f}  The best FDE:{:.6f}'.format(ade_best / scale, best_ans / scale))
            print('The best performance in epoch:'+str(best_epoch))
        print('*'*60)