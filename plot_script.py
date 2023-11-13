import numpy as np
from plotter import plotter

#xx = np.load("res_dir/qgan_nepoch5_N200_depth1.npz")
#xx = np.load("res_dir/qgan_nepoch100_N1000_depth2.npz")
#xx = np.load("res_dir/qgan_nepoch100_N2000_depth2.npz")
#xx = np.load("res_dir/qgan_nepoch100_N1000_depth2_iter2.npz")
#xx = np.load("res_dir/qgan_nepoch100_N1000_depth3_iter0.npz")
#iter_list = [0,1,2,3,4,5]
iter_list = [0]

for ind in range(len(iter_list)):
    iter = iter_list[ind]
    #xx = np.load("res_dir/qgan_nepoch200_N2000_depth1_iter"+str(iter)+".npz")
    #xx = np.load("res_dir/qgan_nepoch100_N1000_depth2_iter" + str(iter) + ".npz")
    #xx = np.load("res_dir/qgan_lr1e3_nepoch100_N1000_depth1_iter" + str(iter) + ".npz")
    #qgan_lr1e3_initdist_nepoch400_N1000_depth2_iter0.npz
    xx = np.load("res_dir/qgan_lr1e3_initdist_nepoch200_N1000_depth2_iter" + str(iter) + ".npz")
    plotter_obj = plotter(xx["num_epochs"])
    plotter_obj.plot_dist(xx['real_data_round'], xx['bounds'], xx['samples_g'], xx['prob_g'], xx['num_dim'])
    plotter_obj.plot_rel_entropy(xx['rel_entr'])
    gloss = xx['g_loss']
    gloss[1:] = (gloss[0:-1]+gloss[1:])/2
    plotter_obj.plot_loss(gloss, xx['d_loss'])

brkpnt1 = 1