import seaborn as sns
import matplotlib.pyplot as plt
from pylab import *
import torch
import random
import pickle
from tools.utils import *
from tools.world2pixel import *
from PIL import Image
import copy
from tqdm import tqdm
def Colourlist_Generator(n):
    Rangelist = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    n = int(n)
    Colours = []             
    j = 1
    while j <= n:            
        colour = ""          
        for i in range(6):
            colour += Rangelist[random.randint(0,14)]    
        colour = "#"+colour                              
        Colours.append(colour)
        j = j+1
    return Colours

def vis_gaussian_distribution(data):

    # data: [20 T N 2]
    data = data.reshape(-1, data.shape[2], data.shape[-1])
    for i in range(data.shape[1]):
        x = data[:,i,0]
        y = data[:,i,1]
        sns.kdeplot(x, y, camp="Reds",shade_lowest=False,shade=True)

def kmeans(data, k, max_time = 100):
	n, m = data.shape
	ini = torch.randint(n, (k,)) 
	midpoint = data[ini]   
	time = 0
	last_label = 0
	while(time < max_time):
		d = data.unsqueeze(0).repeat(k, 1, 1)  
		mid_ = midpoint.unsqueeze(1).repeat(1,n,1) 
		dis = torch.sum((d - mid_)**2, 2)    
		label = dis.argmin(0)      
		if torch.sum(label != last_label)==0:  
			return label        
		last_label = label
		for i in range(k):  #
			kpoint = data[label==i] 
			if i == 0:
				midpoint = kpoint.mean(0).unsqueeze(0)
			else:
				midpoint = torch.cat([midpoint, kpoint.mean(0).unsqueeze(0)], 0)
		time += 1
	return label
sets_names = ['eth','hotel','univ','zara1','zara2']
for sets_name in sets_names:
    with open('vis_data/'+sets_name+'_best.pkl', 'rb') as f:
        data = pickle.load(f)
        print(len(data))
        for i in tqdm(range(len(data))):
            frame = data[i]['frame']
            obs = data[i]['obs'].cpu().numpy()
            gt = data[i]['gt'].cpu().numpy()
            pre = data[i]['pre'].cpu().numpy()
            gt = w2p(gt, sets_name)
            obs = w2p(obs, sets_name)
            for j in range(20):
                pre[j] = w2p(pre[j], sets_name)
            plt.cla()
            im = array(Image.open('pics/'+sets_name+'_newimg/frame'+str(int(frame))+'.jpg'))
            plt.imshow(im)
            for j in range(gt.shape[0]):
                x = obs[j, :, 0]
                y = obs[j, :, 1]
                plt.plot(x,y,color='lime',marker='',linestyle='-',markersize=4)
                label = kmeans(torch.from_numpy(pre[:, j, :, :].reshape(-1, 24)),k=3)
                for k in range(20):
                    x = pre[k,j,:,0]
                    y = pre[k,j,:,1]
                    if label[k] == 0:
                        plt.plot(x,y,color='orange',marker='',linestyle='--',markersize=4,alpha=0.5)
                    if label[k] == 1:
                        plt.plot(x,y,color='green',marker='',linestyle='--',markersize=4)
                    if label[k] == 2:
                        plt.plot(x,y,color='yellow',marker='',linestyle='--',markersize=4,alpha=0.2)
                x = gt[j, :, 0]
                y = gt[j, :, 1]
                plt.plot(x,y,color='aqua',marker='',linestyle='-',markersize=4)
            plt.savefig(sets_name+'/'+str(i)+'.png')