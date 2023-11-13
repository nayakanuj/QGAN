import numpy as np
from plotter import plotter
import matplotlib.pyplot as plt

#iter_list = [0,1,2,3,4,5]
iter_list = [0,1,2]
#iter_list = [0,2]
ckt_depth_arr = np.array([1,2,3,4])
kldiv_arr = []
kldiv_stdarr = []
p_list = [0.0, 0.01, 0.1]
num_epoch_list = [200,100,200]
#p_list = [0.01]
avglen = 10

plt.figure()
for indp in range(len(p_list)):
    p_noise = p_list[indp]
    kldiv_arr = []
    kldiv_stdarr = []
    for ind_depth in range(len(ckt_depth_arr)):
        ckt_depth = ckt_depth_arr[ind_depth]
        kldiv_val = 0.0
        kldiv_all_iter = []
        for ind in range(len(iter_list)):
            iter_num = iter_list[ind]
            #xx = np.load("res_dir/qgan_lr1e3_nepoch100_N1000_depth"+str(ckt_depth)+"_iter" + str(iter_num) + ".npz")
            if p_noise == 0.0:
                xx = np.load("res_dir/qgan_lr1e3_initdist_nepoch200_N1000_depth" + str(ckt_depth) + "_iter" + str(iter_num) + ".npz")
            else:
                xx = np.load("res_dir/qgan_pnoise" + str(p_noise) + "_lr1e3_initdist_nepoch"+str(num_epoch_list[indp])+"_N1000_depth" + str(ckt_depth) + "_iter" + str(iter_num) + ".npz")
            #xx = np.load("res_dir/qgan_pnoise"+str(p_noise)+"_lr1e3_initdist_nepoch200_N1000_depth" + str(ckt_depth) + "_iter" + str(iter_num) + ".npz")
            kldiv_val += np.sum(xx['rel_entr'][-avglen:])
            kldiv_all_iter.append(kldiv_val)

        kldiv_arr.append(kldiv_val/len(iter_list)/avglen)
        kldiv_stdarr.append(np.std(kldiv_all_iter)*0.05)

    #kldiv_arr = np.array(kldiv_arr)
    kldiv_arr = np.array(kldiv_arr)
    kldiv_stdarr = np.array(kldiv_stdarr)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 15

    plt.plot(ckt_depth_arr, kldiv_arr, "-s", label="p="+str(p_list[indp]))
    plt.fill_between(ckt_depth_arr, kldiv_arr-kldiv_stdarr, kldiv_arr+kldiv_stdarr, alpha=0.5)

plt.grid()
plt.xlabel("Circuit depth (k)")
plt.ylabel("$D_{KL}(p_{data} || p_{g})$")
plt.legend()

brkpnt1 = 1