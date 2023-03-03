# %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
import torch
from scipy.stats import multivariate_normal

def gauss_fun(X, Y, data):

    mux = list(data[:,0])
    muy = list(data[:,1])
    sx = list(data[:,2])
    sy = list(data[:,3])
    rho = list(data[:,4])

    # mux = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    # muy = [1.2, 1.7, 2.2, 2.7, 3.2, 3.7]
    # sx = [0.5, 0.6, 0.65, 0.55, 0.7, 0.8]
    # sy = [1.0, 1.2, 1.2, 1.15, 0.9, 0.8]
    # rho = [0.1, 0.2, 0.2, 0.19, 0.2, 0.2]
    
    d = np.dstack([X, Y])
    
    z = None
    
    for i in range(len(mux)):
        mean = [mux[i], muy[i]]

        # Extract covariance matrix
        cov = [[sx[i] * sx[i], rho[i] * sx[i] * sy[i]], [rho[i] * sx[i] * sy[i], sy[i] * sy[i]]]

        gaussian = multivariate_normal(mean = mean, cov = cov)
        
        z_ret = gaussian.pdf(d)
        
        if z is None:
            z = z_ret
            
        else:
            z += z_ret
            
    return z
def gen_gauss(data,traj,dest,sam,gt,p):
    plt.clf()
    x = np.linspace(torch.min(data[:,0]).cpu()-1, torch.max(data[:,0]).cpu()+1, 50)
    y = np.linspace(torch.min(data[:,1]).cpu()-1, torch.max(data[:,1]).cpu()+1, 50)

    
    # x = np.linspace(-3, 3, 40)
    # y = np.linspace(-3, 3, 40)

    X, Y = np.meshgrid(x, y)
    Z = gauss_fun(X, Y, data)
    contours = plt.contour(X, Y, Z, 10, colors='black')
    plt.clabel(contours, inline=True, fontsize=8)
    plt.pcolormesh(X, Y, Z, cmap='RdBu')
    plt.colorbar()
    plt.scatter(gt[0].cpu(),gt[1].cpu(),s=200,c='pink',marker='*')
    for flag in range(20):
        if dest[flag][0] >= (torch.min(data[:,0]).cpu()-1) and dest[flag][0] <= (torch.min(data[:,0]).cpu()+1) \
            and dest[flag][1] >= (torch.min(data[:,1]).cpu()-1) and dest[flag][1] <= (torch.min(data[:,1]).cpu()+1):
            plt.scatter(dest[flag][0].cpu(),dest[flag][1].cpu(),s=60,c='yellow')
    # for flag in range(20):
    #     if sam[flag][0] >= (torch.min(data[:,0]).cpu()-1) and sam[flag][0] <= (torch.min(data[:,0]).cpu()+1) \
    #         and sam[flag][1] >= (torch.min(data[:,1]).cpu()-1) and sam[flag][1] <= (torch.min(data[:,1]).cpu()+1):
    #         plt.scatter(sam[flag][0].cpu(),sam[flag][1].cpu(),s=60,c='aqua')
    plt.savefig('trick_pami_vis/wotrick/'+str(p)+'.jpg')