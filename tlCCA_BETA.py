# -*- coding:utf-8 -*-
'''
Author:
    RuixinLuo  ruixin_luo@tju.edu.cn
Application:
    BETA SSVEP dataset with 40 commands.
    algorithm: tlCCA
'''
import os, sys
import scipy.io as sio
import warnings
warnings.filterwarnings('ignore')
import time
from joblib import Parallel, delayed
import numpy as np
from sklearn.model_selection import ShuffleSplit,LeaveOneOut
from toolbox import PreProcessing_BETA,acc_calculate,ms_eCCA_spatialFilter,decompose_tlCCA,reconstruct_tlCCA_classification,tlCCA_test
warnings.filterwarnings('ignore')


def beta_tlCCA_CrossFre(idx_num, n_train, t_task,A_ind,B_ind):

    # setting
    f_list = np.array([8.6, 8.8,
              9, 9.2, 9.4, 9.6, 9.8,
              10, 10.2, 10.4, 10.6, 10.8,
              11, 11.2, 11.4, 11.6, 11.8,
              12, 12.2, 12.4, 12.6, 12.8,
              13, 13.2, 13.4, 13.6, 13.8,
              14, 14.2, 14.4, 14.6, 14.8,
              15, 15.2, 15.4, 15.6, 15.8,
              8, 8.2, 8.4, ])
    target_order = np.argsort(f_list)
    f_list = f_list[target_order]
    phase_list = np.array([
        0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5,
         0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5,
         0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5,
         0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5,
         0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5,
    ])
    subject_id = ['S'+'{:02d}'.format(idx_subject+1) for idx_subject in range(70)]

    idx_num = idx_num
    idx_subject = subject_id[idx_num]
    sfreq = 250
    filepath = r'Beta'
    filepath = os.path.join(filepath, str(idx_subject) + '.mat')
    num_filter = 5
    preEEG = PreProcessing_BETA(filepath, t_begin=0.5, t_end=0.5 + 0.13 + t_task,  
                           fs_down=250, chans=['POZ', 'PZ', 'PO3', 'PO5', 'PO4', 'PO6', 'O1', 'OZ', 'O2'],
                           num_filter=num_filter)

    raw_data = preEEG.load_data()
    w_pass_2d = np.array([[5, 14, 22, 30, 38, 46, 54, 62], [90, 90, 90, 90, 90, 90, 90, 90]])  
    w_stop_2d = np.array([[3, 12, 20, 28, 36, 44, 52, 60], [92, 92, 92, 92, 92, 92, 92, 92]])  
    filtered_data = preEEG.filtered_data_iir111(w_pass_2d, w_stop_2d, raw_data)

    filtered_data['bank1'] = filtered_data['bank1'][:, : ,target_order,:]  # Sorted by frequency in ascending order
    filtered_data['bank2'] = filtered_data['bank2'][:, :, target_order, :]
    filtered_data['bank3'] = filtered_data['bank3'][:, :, target_order, :]
    filtered_data['bank4'] = filtered_data['bank4'][:, :, target_order, :]
    filtered_data['bank5'] = filtered_data['bank5'][:, :, target_order, :]

    nBlock = 4  #
    train_size = n_train  
    n_splits = 4
    rand_order = np.arange(nBlock)

    # time setting
    t = t_task
    task_point = np.arange(int((0.13) * sfreq), int((0.13 + t) * sfreq))

    # train : SUBSET A
    acc_s = 0
    for k in range(n_splits):
        np.random.seed(k)
        np.random.shuffle(rand_order)
        # train : get model of banks
        train_U = dict()
        train_V = dict()
        train_W = dict()
        train_Hr = dict()
        for idx_filter in range(num_filter):
            idx_filter += 1
            bank_data = filtered_data['bank' + str(idx_filter)]
            train_data11 = bank_data[:, :, :, rand_order[:train_size]]  # randomly selected train trials
            train_data = train_data11[:, task_point, :, :]  # n_channels, n_times, n_events, n_trials
            train_data = train_data[:, :, A_ind, : ] # subset A
            # train
            Nh=5
            U = np.zeros((9, 20))
            V = np.zeros((Nh * 2, 20))
            for iEvent in range(20):
                # msCCA: u,v
                u, v = ms_eCCA_spatialFilter(np.mean(train_data,-1), iEvent=iEvent,
                                             nTemplates=12, fs=sfreq, f_list=f_list[A_ind], phi_list=phase_list[A_ind], Nh=Nh)
                U[:, iEvent] = u
                V[:, iEvent] = v
            # Decompose SSVEPs to the subset A stimulus
            w_all, r_all = decompose_tlCCA(mean_temp=np.mean(train_data,-1), fre_list=f_list[A_ind], ph_list=phase_list[A_ind],
                                           Fs=sfreq, Oz_loc=7)
            # Reconstruct SSVEPs to the subset B stimulus
            Hr = reconstruct_tlCCA_classification(r=r_all, fre_list_target=f_list[B_ind], fre_list_souce=f_list[A_ind],
                                   ph_list=phase_list[B_ind], Fs=sfreq, tw=t_task)
            #
            train_U['bank' + str(idx_filter)] = U  # n_channels, n_events/2
            train_V['bank' + str(idx_filter)] = V  # 2 * n_harmonics, n_events/2
            train_W['bank' + str(idx_filter)] = np.array(w_all).T # n_channels, n_events/2
            train_Hr['bank' + str(idx_filter)] = Hr # n_times, n_events/2

        # test: SUBSET B
        predictAll = np.zeros((nBlock, 20))
        flag = 0
        for isplit in np.arange(nBlock): # test all blocks in subset B
            rrall = np.zeros((20, 20))
            for idx_filter in range(num_filter):
                idx_filter += 1
                bank_data = filtered_data['bank' + str(idx_filter)]
                test_data = bank_data[:, :, :, isplit]
                test_data = test_data[:, task_point, :]
                test_data = test_data[:, :, B_ind]  # subset B
                rr = tlCCA_test(test_data, train_Hr['bank' + str(idx_filter)] ,
                                 train_U['bank' + str(idx_filter)] ,train_V['bank' + str(idx_filter)] ,train_W['bank' + str(idx_filter)] ,fs=sfreq,
                                 f_list=f_list[B_ind] , phi_list=phase_list[B_ind] ,Nh=5)
                rrall += np.multiply(np.sign(rr), (rr ** 2)) * (idx_filter ** (-1.25) + 0.25)
            predict = np.argmax(rrall, -1)
            predictAll[flag, :] = predict
            flag += 1
        acc_s = acc_calculate(predictAll) + acc_s
    acc = acc_s / n_splits
    print('sub', idx_num + 1, ', acc = ', acc_s / n_splits)

    return acc


if __name__ == '__main__':

    A = np.arange(0, 40, 2)  # A dataset 0,2,4,...38
    B = np.arange(1, 40, 2)  # B dataset 1,3,5,...39

    # acc = beta_tlCCA_CrossFre(idx_num=9, t_task=1 ,n_train=3, A_ind=A,B_ind=B)
    acc_all = np.zeros((70,4))
    for i_train in range(4):
        print('train_size=',i_train+1)
        acc = Parallel(n_jobs=1)(delayed(beta_tlCCA_CrossFre)(idx_num, n_train=i_train+1, t_task=1, A_ind=A,B_ind=B) for idx_num in range(70))
        acc =  np.array(acc)
        acc_all[:,i_train] = acc
        print('mean_acc=',np.mean(acc,-1))
    sio.savemat(r'tlCCA_acc_BETA.mat', {'acc': acc_all})
    # if everything is ok, You will get the following output:
    # when t_task=0.5s ,train_size=4 , the mean_acc is 0.6554 
    # when t_task=1s ,train_size=4 , the mean_acc is 0.8550
    # Note that this result is almost the same as that of Fig.7 in the paper









