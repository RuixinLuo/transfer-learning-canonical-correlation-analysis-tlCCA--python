# -*- coding:utf-8 -*-
'''
Author:
    RuixinLuo  ruixin_luo@tju.edu.cn
Application:
    Benchmark SSVEP dataset with 40 commands.
    we decompose an SSVEP corresponding to the i-th stimulus and then reconstruct a new SSVEP corresponding to the (i+1)-th stimulus.
    source code: https://github.com/edwin465/SSVEP-Impulse-Response/blob/main/demo2a_ssvep_decomposition_reconstruction.m
'''

# load modules
import sys
import numpy as np
import scipy.io as io
import os
from toolbox import PreProcessing,decompose_tlCCA,reconstruct_tlCCA
import matplotlib.pyplot as plt


if __name__ == '__main__':

    ycor_all = []
    ymse_all = []
    # setting
    freq_tmp = np.arange(8, 16, 1)
    f_list = np.hstack((freq_tmp, freq_tmp + 0.2, freq_tmp + 0.4, freq_tmp + 0.6, freq_tmp + 0.8))
    target_order = np.argsort(f_list)
    f_list = f_list[target_order]
    phase_list = np.array([
        0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5,
         0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5,
         0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5,
         0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5,
         0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5,
    ])
    subject_id =  ['S'+'{:02d}'.format(idx_subject+1) for idx_subject in range(35)]


    L = []
    sub_id = 0
    for id in subject_id:
        # load data
        sfreq = 250
        fs = sfreq
        filepath = r'Bench'
        filepath = os.path.join(filepath, str(id) + '.mat')
        num_filter = 1
        t_task = 2
        preEEG = PreProcessing(filepath, t_begin=0.5 + 0.14, t_end=0.5 + 0.14 + t_task,  
                               fs_down=250, chans=['POZ', 'PZ', 'PO3', 'PO5', 'PO4', 'PO6', 'O1', 'OZ', 'O2'],
                               num_filter=num_filter)

        raw_data = preEEG.load_data()
        w_pass_2d = np.array([[6, 14, 22, 30, 38, 46, 54, 62], [90, 90, 90, 90, 90, 90, 90, 90]])  
        w_stop_2d = np.array([[4, 12, 20, 28, 36, 44, 52, 60], [92, 92, 92, 92, 92, 92, 92, 92]])  
        filtered_data = preEEG.filtered_data_iir111(w_pass_2d, w_stop_2d, raw_data)
        # io.savemat(r'sub1_sb1.mat', {'data': np.transpose(filtered_data['bank1'],[0,1,3,2])})

        data_pre = filtered_data['bank1'][:, :t_task*sfreq, target_order, :]  # n_channels * n_times * n_event * n_trials
        del raw_data, w_pass_2d, w_stop_2d ,filtered_data

        A = np.arange(0,40,2)  # A dataset 0,2,4,...38
        B = np.arange(1,40,2)  # B dataset 1,3,5,...39
        data_A = data_pre[...,A,:]
        data_B = data_pre[...,B,:]

        # Decompose SSVEPs to the (i)-th stimulus
        w_all, r_all = decompose_tlCCA(mean_temp=np.mean(data_A,-1), fre_list=f_list[A] , ph_list=phase_list[A], Fs=fs, Oz_loc=7)
        # Reconstruct SSVEPs to the (i+1)-th stimulus
        Hr = reconstruct_tlCCA(r= r_all, fre_list_target=f_list[B] ,fre_list_souce=f_list[A], ph_list=phase_list[B], Fs=fs, tw=t_task)

        # compare with Hr and Xw
        ymse = []
        ycor = []
        fig, ax = plt.subplots(4, 5, constrained_layout=True,dpi=600)
        fig.suptitle(id, fontsize=7)
        for i in range((20)):
            # reconstructed signal
            hr = Hr [:,i]
            # real signal after w
            w = w_all[i]
            xw = w @ np.mean(data_B, -1)[:,:,i]
            xw  = xw  - np.mean(xw )
            xw  = xw  / np.std(xw,ddof=1)
            # count corr and mse
            ycor.append(np.corrcoef(hr,xw)[0,1])
            ymse.append(np.linalg.norm(xw-hr)/hr.shape[0])
            # draw fig
            x = np.int(i/5)
            y = i-5*x
            t=np.arange(np.int(0.5*fs))/fs
            ax[x][y].plot(t,hr[:np.int(0.5*fs)], color='b',linewidth=0.5)
            ax[x][y].plot(t,xw[:np.int(0.5*fs)], color='r',linewidth=0.5)
            ax[x][y].set_title(str(f_list[B][i]), fontsize=6)
            # ax[x][y].set_xlabel('times(s)', fontsize=6)
            # ax[x][y].set_ylabel('μV', fontsize=6)
            ax[x][y].tick_params(labelsize=6)
        fig.show()
        ycor = np.array(ycor)
        ycor_all.append(ycor)
        ymse = np.array(ymse)
        ymse_all.append(ymse)
        print(id,';mean_cor=',np.mean(ycor,-1),';mean_ymse=',np.mean(ymse,-1))
        sub_id = sub_id+1
    ycor_all = np.array(ycor_all)
    ymse_all = np.array(ymse_all)
    # the result (ycor_all, ymse_all) is same as the source code (demo2a) with the same pre-processed data (data_pre)



