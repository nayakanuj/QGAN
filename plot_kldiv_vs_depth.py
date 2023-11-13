import numpy as np
from plotter import plotter
import matplotlib.pyplot as plt

iter_list = [0,1,2,3,4,5]
#iter_list = [0,1,2,3,4]
ckt_depth_arr = np.array([1,2,3,4])
kldiv_arr = []
kldiv_stdarr = []

for ind_depth in range(len(ckt_depth_arr)):
    ckt_depth = ckt_depth_arr[ind_depth]
    kldiv_val = 0.0
    kldiv_all_iter = []
    for ind in range(len(iter_list)):
        iter_num = iter_list[ind]
        #xx = np.load("res_dir/qgan_lr1e3_nepoch100_N1000_depth"+str(ckt_depth)+"_iter" + str(iter_num) + ".npz")
        xx = np.load("res_dir/qgan_lr1e3_initdist_nepoch200_N1000_depth" + str(ckt_depth) + "_iter" + str(iter_num) + ".npz")
        kldiv_val += xx['rel_entr'][-1]
        kldiv_all_iter.append(xx['rel_entr'][-1])

    kldiv_arr.append(kldiv_val/len(iter_list))
    kldiv_stdarr.append(np.std(kldiv_all_iter))

#kldiv_arr = np.array(kldiv_arr)
kldiv_arr = np.array(kldiv_arr)
kldiv_stdarr = np.array(kldiv_stdarr)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 15

plt.figure()
plt.plot(ckt_depth_arr, kldiv_arr, "-s")
plt.fill_between(ckt_depth_arr, kldiv_arr-kldiv_stdarr, kldiv_arr+kldiv_stdarr, alpha=0.5)
plt.grid()
plt.xlabel("Circuit depth (k)")
plt.ylabel("$D_{KL}(p_{data} || p_{g})$")

brkpnt1 = 1