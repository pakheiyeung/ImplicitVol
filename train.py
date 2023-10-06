# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:32:59 2019

@author: pemb5552
"""
import torch
from torch.utils import data
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import glob
import os
import numpy as np
from random import shuffle
from tensorboardX import SummaryWriter
import random
import matplotlib.pyplot as plt
import imageio
from ssim import SSIM
import nibabel as nib
import time

from model import LearnPose, OfficialNerf_siren, encode_position, sample_from_matrix
from dataset import Dataset_volume_video, eulerAnglesToRotationMatrix



if __name__ ==  '__main__':
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    np.random.seed(10)
    random.seed(10)
    
    'hyperparameters'
    N_EPOCH = 10001  # training epoch for each set of images
    EVAL_INTERVAL = 50  
    N_IMGS = 128
    trainable = True    # trainable pose
    
    vol_folder = './example/*.nii.gz'
    params_dataset =  {'set_size': N_IMGS,
                       'mode':'training',
                       }
    params = {'batch_size': 1,
              'shuffle': True,
              'num_workers': 4,
              'drop_last':False}
    
    'Iterate each file'
    files = glob.glob(vol_folder)
    for file in files:
        'Initialize dataset and generator'
        training_set = Dataset_volume_video(file, **params_dataset)
        training_generator = data.DataLoader(training_set, **params)
        grid_ref = torch.from_numpy(training_set._sampling_grid_ref()).float().cuda()
        
        'save path'
        
        
        'Initialise all trainabled parameters'
        pose_param_net = LearnPose(training_set._overall_store(), learn_R=trainable, learn_t=trainable).cuda()
        
        'NeRF'
        nerf_model = OfficialNerf_siren(pos_in_dims=63, D=128).cuda()
        
        'SSIM Loss'
        ssim_loss = SSIM(window_size=5)
        
        'Set lr and scheduler: these are just stair-case exponantial decay lr schedulers.'
        opt_nerf = torch.optim.Adam(nerf_model.parameters(), lr=0.001)
        opt_pose = torch.optim.Adam(pose_param_net.parameters(), lr=0.001)
        
        from torch.optim.lr_scheduler import MultiStepLR
        scheduler_nerf = MultiStepLR(opt_nerf, milestones=list(range(0, 10000, 10)), gamma=0.9954)  #10 0.9954
        scheduler_pose = MultiStepLR(opt_pose, milestones=list(range(0, 10000, 100)), gamma=0.9)  #100 0.9
        
        for e in range(N_EPOCH):
            torch.cuda.synchronize() 
            start_epoch = time.time()
            
            'training'
            nerf_model.train()
            pose_param_net.train()
            
            if e<5000 and e%500==0:
                nerf_model = OfficialNerf_siren(pos_in_dims=63, D=128).cuda()
                opt_nerf = torch.optim.Adam(nerf_model.parameters(), lr=0.001)
                opt_pose = torch.optim.Adam(pose_param_net.parameters(), lr=0.001)
                
                scheduler_nerf = MultiStepLR(opt_nerf, milestones=list(range(0, 10000-e, 10)), gamma=0.9954)  #10 0.9954
                scheduler_pose = MultiStepLR(opt_pose, milestones=list(range(0, 10000-e, 100)), gamma=0.9)  #100 0.9
            
            
            loss_epoch=[]
            for (local_key, local_img, _,_,_,_) in training_generator:
                local_img = local_img.to(dtype=torch.float).cuda()
                _, H, W = local_img.size()
                key = (local_key.numpy())
                
                pos_enc = []
                for k in key:
                    rot, trans = pose_param_net(k)
                    grid = sample_from_matrix(grid_ref, rot, trans) #HW3
                
                    p = encode_position(grid/(H//2), levels=10, inc_input=True)
                    pos_enc.append(p)
                pos_enc = torch.stack(pos_enc, dim=0)
                
                pred = nerf_model(pos_enc)
                # pred = nn.ReLU()(pred)
                
                # loss = F.mse_loss(pred.squeeze(-1), local_img)   #L2
                loss = -ssim_loss(pred.squeeze(-1).unsqueeze(1), local_img.unsqueeze(1))    #ssim
                
                loss.backward()
                opt_nerf.step()
                opt_pose.step()
                opt_nerf.zero_grad()
                opt_pose.zero_grad()
        
                loss_epoch.append(loss)
                
            torch.cuda.synchronize()
            end_epoch = time.time()
            elapsed = end_epoch - start_epoch
            loss_epoch_mean = torch.stack(loss_epoch).mean().item()
            if (e) % 1 == 0:
                print(e, loss_epoch_mean, elapsed)
            scheduler_nerf.step()
            scheduler_pose.step()
            
            
            'testing - sample orthogonal slices from nerf'
            with torch.no_grad():
                if (e+1) % EVAL_INTERVAL == 0:
                   
                    fig = plt.figure(figsize=(10, 5))
                    ax_img = fig.add_subplot(121)
                    ax_atl_pred = fig.add_subplot(122)
                    gif_frames = []
                    
                    overall_store_test = training_set.return_test()
                    for i in overall_store_test.keys():
                        img = overall_store_test[i]['img']
                        rot = overall_store_test[i]['rot_ground']
                        trans = overall_store_test[i]['trans_ground']
                        rot = eulerAnglesToRotationMatrix(rot)
                        grid = sample_from_matrix(grid_ref, (torch.from_numpy(rot)).float().cuda(), (torch.from_numpy(trans)).float().cuda()) #HW3
                        
                        pos_enc = encode_position(grid/(H//2), levels=10, inc_input=True)
                
                        pred = nerf_model(pos_enc)
                        # pred = nn.ReLU()(pred)
                        
                        'plot and show gif'
                        ax_atl_pred.cla()
                        ax_atl_pred.imshow(pred.detach().squeeze().cpu().numpy(), cmap='gray')
                        ax_atl_pred.axis('off')
                      
                        # plot plane
                        ax_img.cla()
                        ax_img.imshow(img, cmap='gray')
                        ax_img.set_title(i)
                        ax_img.axis('off')
                        plt.pause(0.01)
                        
                    
                        # create gif
                        fig.canvas.draw()       # draw the canvas, cache the renderer
                        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        gif_frames.append(image)
                    
                        
                    'save model'
                    # model_path = os.path.join(gif_folder, '{}.pth'.format(e))
                    # torch.save(nerf_model.state_dict(), model_path)
                    
                    # if save_gif:
                    # if e in [5000, 10000]:
                    #     imageio.mimsave(os.path.join(gif_folder, '{}.gif'.format(e+int(train_psnr))), gif_frames, fps=8)
                    plt.close('all')
            
            