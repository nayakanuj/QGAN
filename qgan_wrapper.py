
from qgan_top import qgan_top
from qgan_top_real_hw import qgan_top_real_hw
from qgan_top_noisy import qgan_top_noisy
import matplotlib.pyplot as plt

num_epochs = 200
dataset_size = 1000
#iter_list = [0,1,2,3,4,5]
#iter_list = [0,1,2]
iter_list = [0]
#num_epochs=20
#dataset_size=200
#num_iters = 2
#ckt_depth_list = [1,2,3,4]
ckt_depth_list = [1]
#ckt_depth_list = [3]
#ckt_depth_list = [4]
p_noise = 0.1
for ind in range(0, len(iter_list)):
    iter_num = iter_list[ind]
    for ind_ckt_depth in range(len(ckt_depth_list)):
        ckt_depth = ckt_depth_list[ind_ckt_depth]
        print(f"nepochs={num_epochs}, N={dataset_size}, depth={ckt_depth}, iter={iter_num}")
        #save_filename = "res_dir/qgan_lr1e3_initdist_nepoch"+str(num_epochs)+"_N"+str(dataset_size)+"_depth"+str(ckt_depth)+"_iter"+str(iter_num)+".npz"
        #qgan_top(num_epochs=num_epochs, dataset_size=dataset_size, ckt_depth=ckt_depth, save_filename=save_filename)
        save_filename = "res_dir/qgan_pnoise"+str(p_noise)+"_lr1e3_initdist_nepoch" + str(num_epochs) + "_N" + str(dataset_size) + "_depth" + str(ckt_depth) + "_iter" + str(iter_num) + ".npz"
        qgan_top_noisy(num_epochs=num_epochs, dataset_size=dataset_size, ckt_depth=ckt_depth, save_filename=save_filename, p_noise = p_noise)
        plt.close('all')


# # Sanity check
# num_epochs=5
# dataset_size=200
# ckt_depth=1
# save_filename = "qgan_nepoch"+str(num_epochs)+"_N"+str(dataset_size)+"_depth"+str(ckt_depth)+".npz"
# qgan_top(num_epochs=num_epochs, dataset_size=dataset_size, ckt_depth=ckt_depth, save_filename=save_filename)
