> # transfer learning canonical correlation(tlCCA) for SSVEPs -python

#### transfer learning canonical correlation(tlCCA) based on paper [1]_.

Referenced code available from https://github.com/edwin465/SSVEP-Impulse-Response and https://github.com/edwin465/SSVEP-tlCCA, from MATLAB to python

> [1] Wong, C. M., et al. (2021). Transferring Subject-Specific Knowledge Across Stimulus Frequencies in SSVEP-Based BCIs. IEEE Transactions on Automation Science and Engineering. (Accepted)

- #### demo1a:  

  we decompose an SSVEP and then reconstruct it (corresponding to the same stimulus).

  source code: https://github.com/edwin465/SSVEP-Impulse-Response/blob/main/demo1a_ssvep_decomposition_reconstruction.m

- #### demo2a:  

  we decompose an SSVEP corresponding to the i-th stimulus and then reconstruct a new SSVEP corresponding to the (i+1)-th stimulus.

  source code: https://github.com/edwin465/SSVEP-Impulse-Response/blob/main/demo2a_ssvep_decomposition_reconstruction.m

- #### tlCCA_BETA:

  The performance of tlCCA was tested on BETA dataset. Specifically, all stimulus frequencies were divided into set A and set B. The spatial filters and impulse responses constructed from the training data in set A were used to test the data in set B.

  

## Dataset

#### Benchmark dataset [2]_ from Tsinghua university.

>  [2] Wang Y , Chen X , Gao X , et al. A Benchmark Dataset for SSVEP-Based Brain-Computer Interfaces[J].IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2017, 25(10):1746-1752.

#### BETA dataset [3]_ from Tsinghua university.

> [3] Liu B ,  Huang X ,  Wang Y , et al. BETA: A Large Benchmark Database Toward SSVEP-BCI Application[J]. Frontiers in Neuroscience, 2020, 14.

## results

- #### demo1a:  

the result (ycor_all, ymse_all) is same as that in the source code (demo1a), when the input preprocessed data is the same(data_pre).

- #### demo2a:  

the result (ycor_all, ymse_all) is same as  that in the source code (demo2a) , when the input preprocessed data is the same(data_pre).

- #### tlCCA_BETA:

if everything is ok, You will get the following output:

when t_task=0.5s ,train_size=4 , the mean_acc is 0.6554 

when t_task=1s ,train_size=4 , the mean_acc is 0.8550

Note that  the result when t_task=1s is almost the same as that of source code (https://github.com/edwin465/SSVEP-tlCCA), but the  result when t_task=0.5s is lower than that of source code. 

In fact, in the source code, the data of  2 s time length in set A is always used to obtain the transferred spatial filters and templates. However, here we applied the same signal length as set B used for testing (<2 s), so it is normal that the performance of tlCCA_BETA in the short time window is lower than that of the source code.

we think that, to explore semi-supervised learning for SSVEP-BCIs, it is better to have equal time lengths for SetA and SetB.

## Acknowledgement

Thanks to [Chi Man Wong]([edwin465 (Chi Man Wong) (github.com)](https://github.com/edwin465)), the author of tlCCA, for his patience in responding to my questions.

## email

ruixin_luo@tju.edu.cn



