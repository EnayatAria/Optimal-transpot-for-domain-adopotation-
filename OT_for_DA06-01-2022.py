'''
This code is for applying optimal transport as a domain adaptation algorithm to remote sensing images
Four different OT functions is used in this code: Earth moving distance (EMD), Sinkhorn, Group Sparsity, and Laplace
I used the inverse transformation in this code: i.e. target data is mapped into source data

Input:
    - source image dataset in envi format
    - target image dataset in envi format
    - training samples of the classes of source dataset
Output:
    - four transformed image sets using  four OT functions in envi format
    - Plot of the point distributions of the source and target data and the transformed target images

developer: Enayat Aria
06/01/2022
'''

# for not having the cursor problem
# !/usr/bin/env python
import os
import time
from datetime import datetime
from typing import TextIO

import matplotlib.patches as mpatches
import matplotlib.pylab as pl
import numpy as np
import ot
import spectral.io.envi as envi
from matplotlib.colors import ListedColormap

from numpy import ndarray


def entire_Tdata_trans(scene_name, tf_name, Data_target1, ot_base_tr,
                       lambda_reg, regCL_gs, regCL_lap, ot_v, mode, savepath):
    '''
    This function is to transform an entire target scene using the given OT model
    :param tf_name:
    :param Data_targe1:
    :param ot_base_tr:
    :return:
    '''
    Xt_whole = np.reshape(Data_target1, (Data_target1.shape[0] * Data_target1.shape[1], Data_target1.shape[2]))
    transp_Xt_whole = ot_base_tr.inverse_transform(Xt=Xt_whole)  # identify the OT method
    # reshape target image
    transp_Xt_reshp = np.reshape(transp_Xt_whole, (Data_target1.shape[0], Data_target1.shape[1], 5))
    transp_Xt_reshp: ndarray = transp_Xt_reshp.astype('float32')
    # dd/mm/YY H:M:S
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    md = img2.metadata
    if ot_v == 'ts':
        ot_version = 'transformation using training sites'
    else:
        ot_version = 'transformation using classification map'
    if mode == 3:
        no_class = 'without grass transformation (3 classes)'
    else:
        no_class = 'with grass transformation (4 classes)'

    md[
        'description'] = tf_name + ';' + dt_string + '\n' \
                         + ot_version + '; ' + no_class + '\n' \
                         + 'ot.da.EMDTransport() \n' \
                         + 'ot.da.SinkhornTransport(reg_e=' + str(lambda_reg) + ', max_iter=20000, verbose=True) \n' \
                         + 'ot.da.SinkhornL1l2Transport(reg_e=' + str(lambda_reg) + ', reg_cl= ' + str(regCL_gs) + \
                         ', max_iter=100, max_inner_iter=20000, verbose=True)  \n' \
                         + 'ot.da.EMDLaplaceTransport(reg_lap=' + str(regCL_lap) + \
                         ', reg_src=1, max_iter=200, verbose=True, max_inner_iter=150000) \n '

    md['data type'] = 4
    envi.save_image(savepath + tf_name + '_lambda_'+str(lambda_reg)+'_regGS_'+str(regCL_gs)+'.hdr', transp_Xt_reshp,
                    dtype=np.float32, force=True, metadata=md)


def plot_transport(scene_name, ot_v, mode, savepath):
    """this function plots the distribution of the source, target, and transported target using various OT types"""
    fig = pl.figure(1, figsize=(11, 8))

    ax1 = pl.subplot(2, 3, 1)
    b1 = 1
    b2 = 4
    data_x = [[Xs[:, b1]], [Xt[:, b1]], [transp_Xt_emd[:, b1]], [transp_Xt_sinkhorn[:, b1]], [transp_Xt_l1l2[:, b1]],
              [transp_Xt_emd_laplace[:, b1]]]
    data_y = [[Xs[:, b2]], [Xt[:, b2]], [transp_Xt_emd[:, b2]], [transp_Xt_sinkhorn[:, b2]], [transp_Xt_l1l2[:, b2]],
              [transp_Xt_emd_laplace[:, b2]]]

    if ot_v == 'ts':
        ot_ver = 'transformation using training sites'
    else:
        ot_ver = 'transformation using classification map'
    if mode == 3:
        no_class = 'without grass (3 classes)'
        cmap = ListedColormap(["red", "lightgreen", "darkgreen"])
        label = ["Soil", "Vine", "Shadow"]
    else:
        no_class = 'with grass (4 classes)'
        cmap = ListedColormap(["red", "lightgreen", "darkgreen", "yellow"])
        label = ["Soil", "Vine", "Shadow", "Grass"]

    fig.suptitle(scene_name + '; ' + ot_ver + '; ' + no_class)
    mi1 = 1e6
    mi2 = 1e6
    ma1 = 0
    ma2 = 0
    for el in data_x:
        mi = np.min(el)
        ma = np.max(el)
        if mi < mi1: mi1 = mi
        if ma > ma1: ma1 = ma

    for el in data_y:
        mi = np.min(el)
        ma = np.max(el)
        if mi < mi2: mi2 = mi
        if ma > ma2: ma2 = ma

    mi1 = mi1 - (ma1 - mi1) / 20
    mi2 = mi2 - (ma2 - mi2) / 20
    ma1 = ma1 + (ma1 - mi1) / 20
    ma2 = ma2 + (ma2 - mi2) / 20
    ax1.set_xlim(left=mi1, right=ma1)
    ax1.set_ylim(bottom=mi2, top=ma2)
    ax1.scatter(Xs[:, b1], Xs[:, b2], c=ys, cmap=cmap, marker='+')
    pl.xlabel('band ' + str(b1 + 1))
    pl.ylabel('band ' + str(b2 + 1))
    # pl.yticks([])
    ax1.legend(loc=0)
    pl.title('Source  samples')
    patches = [mpatches.Patch(color=cmap.colors[i], label=label[i]) for i in range(len(label))]
    pl.legend(handles=patches, bbox_to_anchor=(.65, 0), loc='lower left', borderaxespad=0.2,
              fontsize='small')  # bbox_to_anchor=(.85, 0), loc='lower left',

    ax2 = pl.subplot(2, 3, 2)
    ax2.scatter(Xt[:, b1], Xt[:, b2], c='orange', marker='+')
    ax2.set_xlim(left=mi1, right=ma1)
    ax2.set_ylim(bottom=mi2, top=ma2)
    # pl.xticks([])
    # pl.yticks([])
    pl.legend(loc=0)
    pl.title('Target samples')
    pl.tight_layout()
    pl.xlabel('band ' + str(b1 + 1))
    pl.legend().remove()

    ax3 = pl.subplot(2, 3, 3)
    ax3.scatter(transp_Xt_emd[:, b1], transp_Xt_emd[:, b2], c='blue', marker='+')
    ax3.set_xlim(left=mi1, right=ma1)
    ax3.set_ylim(bottom=mi2, top=ma2)
    pl.legend(loc=0)
    pl.title('Target samples transformed by EMD')
    pl.tight_layout()
    pl.legend().remove()

    ax4 = pl.subplot(2, 3, 4)
    ax4.scatter(transp_Xt_sinkhorn[:, b1], transp_Xt_sinkhorn[:, b2], c='blue', marker='+')
    ax4.set_xlim(left=mi1, right=ma1)
    ax4.set_ylim(bottom=mi2, top=ma2)
    pl.xlabel('band ' + str(b1 + 1))
    pl.ylabel('band ' + str(b2 + 1))
    pl.legend(loc=0)
    pl.title('Target samples transformed by Sinkhorn')
    pl.legend().remove()

    ax5 = pl.subplot(2, 3, 5)
    ax5.scatter(transp_Xt_l1l2[:, b1], transp_Xt_l1l2[:, b2], c='blue', marker='+')
    ax5.set_xlim(left=mi1, right=ma1)
    ax5.set_ylim(bottom=mi2, top=ma2)
    pl.xlabel('band ' + str(b1 + 1))
    pl.legend(loc=0)
    pl.title('Target samples transformed by GS')
    pl.legend().remove()

    ax6 = pl.subplot(2, 3, 6)
    ax6.scatter(transp_Xt_emd_laplace[:, b1], transp_Xt_emd_laplace[:, b2], c='blue', marker='+')
    ax6.set_xlim(left=mi1, right=ma1)
    ax6.set_ylim(bottom=mi2, top=ma2)
    pl.xlabel('band ' + str(b1 + 1))
    pl.legend(loc=0)
    pl.title('Target samples transformed by LP')

    pl.tight_layout()
    pl.legend().remove()
    pl.savefig(savepath + 'tf_plot_best.jpeg', dpi=400)
    # pl.savefig('//10.4.0.1/d$/Aria-data/Minervois_2016/' + scene_name[:6] + scene_name[
    #    -1] + '/without Grass transformation/' + 'tf_plot_TS_1600_3'
    #                                             '.jpeg', dpi=400)
    pl.show()


start_time = time.time()
# reading the image data and ground truth
DataPath_source = '//10.4.0.1/d$/Aria-data/Gaillac-UAV-ML-2016/Gaillac-UAV-ML-2016-subset2/Gaillac-UAV-ML-2016-subset2'
# 'C:/Users/aria/Documents/EtSaViTe-Aria/Gaillac-UAV-ML-2016-subset2/Gaillac-UAV-ML-2016-subset2'
DataPath_classification = '//10.4.0.1/d$/Aria-data/Gaillac-UAV-ML-2016/Gaillac-UAV-ML-2016-subset2/pred_all'
# 'C:/Users/aria/Documents/EtSaViTe-Aria/Gaillac-UAV-ML-2016-subset2/pred_all'
txtPath_ROI_source = '//10.4.0.1/d$/Aria-data/Gaillac-UAV-ML-2016/Gaillac-UAV-ML-2016-subset2/4_class_roi.txt'

# Reading envi dataset

img1 = envi.open(DataPath_source + '.hdr', DataPath_source)
Data_source = img1.load()
# converting the source data to a vector
Ds_rs = np.reshape(Data_source, (Data_source.shape[0] * Data_source.shape[1], Data_source.shape[2]))

img3 = envi.open(DataPath_classification + '.hdr', DataPath_classification + '.img')
Data_classification = img3.load()

'''
# standardization of source data
mn = np.mean(np.mean(Data_source, axis=0), axis=0)
sdv = np.std(np.std(Data_source, axis=0), axis=0)
Data_source = (Data_source - mn[None, None, :]) / sdv[None, None, :]
'''
OT_version = ['ts', 'cm'] # ts : training site ; cm : classification map
# OT_version = ['cm']
Lambda_mat = np.array(
    [[190000, 440000, 990000, 740000], [590000, 190000, 490000, 390000], [90000, 340000, 990000, 940000]])
RegCL_gs_mat = np.array([[1000000, 0.01, 1000000, 100000], [0.01, 0.01, 1000000, 0.01], [1000000, 100000, 100000, 1000000]])
RegCL_lap_mat = np.array(
    [[100000, 100000, 100000, 100000], [100000, 1000, 100000, 100000], [100, 10000, 100000, 10000]])

# the optimal lambda values in different situations

for ot_v in OT_version:
    if ot_v == 'ts':
        ''' using RIO (pure samples already selected per class) for the transformation '''
        # making ROI map
        f: TextIO = open(txtPath_ROI_source, 'r')
        line_no = 0
        cls_no: int = 0
        for line in f:
            if 'File Dimension' in line:
                dim = line[18:]
                dim = dim.split(sep='x')
                x_dim = int(dim[0])
                y_dim = int(dim[1])
                ROI_arr = np.zeros([y_dim, x_dim])
            # line_no += 1
            if line[0] != ';' and line[0] != '\n':
                # set_trace()
                if line[5:8] == ' 1 ':
                    cls_no += 1
                ROI_arr[int(line[15:19]), int(line[9:13])] = cls_no
        f.close()
        # Converting data to source and its labels

        roi = np.reshape(ROI_arr, ROI_arr.shape[0] * ROI_arr.shape[1])
        del ROI_arr
    else:
        ''' Selection of the source samples from the classification map '''
        roi = np.reshape(Data_classification, Data_classification.shape[0] *
                         Data_classification.shape[1])  # this is the classification map

    for mode in [3, 4]:  # 3 : without grass (3 classes), 4 : with grass (4 classes)
        # creating the source and its label vector
        for i in range(mode):
            c1 = np.squeeze(np.argwhere(roi == i + 1))
            c2 = int(np.round(len(c1) / 400))
            c3_ys = roi[c1[::c2]]
            c3_xs = Ds_rs[c1[::c2], :]
            if i == 0:
                Xs = c3_xs
                ys = c3_ys
            else:
                ys = np.append(ys, c3_ys)
                Xs = np.append(Xs, c3_xs, axis=0)

        b = np.argwhere(Xs <= 0)
        Xs = np.delete(Xs, b[:, 0], axis=0)
        ys = np.delete(ys, b[:, 0], axis=0)
        (unique, counts) = np.unique(ys, return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        print(frequencies)

        '''
        a_file = open("", "w")
        for row in Xs:
            np.savetxt(a_file, row)
        a_file.close()
        
        a_file = open("Z:/Aria-data/ys.txt", "w")
        np.savetxt("Z:/Aria-data/ys.txt", ys)
        '''


        # del Data_source, Data_classification, img1, img3
        #Scene_List = ['subset_1']
        Scene_List = ['subset_1', 'subset_2', 'subset_3']
        # OT_type = ['EMD', 'Sinkhorn', 'L1L2', 'Laplace']
        OT_type = ['L1L2']

        for scene in Scene_List:
            DataPath_target = '//10.4.0.1/d$/Aria-data/Minervois_2016/' + scene[:6] + scene[-1] + '/' + scene
            img2 = envi.open(DataPath_target + '.hdr', DataPath_target)  # + '.img')
            Data_target = img2.load()

            # creating the target vector
            Xt = np.reshape(Data_target, (Data_target.shape[0] * Data_target.shape[1], Data_target.shape[2]))
            Xt = Xt[::int(np.floor(Xt.shape[0] / 1600)), :]
            a = np.argwhere(Xt <= 0)
            Xt = np.delete(Xt, a[:, 0], axis=0)

            # writing Xt to a text file
            # a_file = open("Z:/Aria-data/Xt.txt", "w")
            # for row in Xt:
            #    np.savetxt(a_file, row)
            # a_file.close()
            if scene == 'subset_1':
                lambda_mat_row = 0
            elif scene == 'subset_2':
                lambda_mat_row = 1
            elif scene == 'subset_3':
                lambda_mat_row = 2
            if ot_v == 'ts' and mode == 4:
                lambda_mat_col = 0
                folder = 'with Grass transformation using training site'
            elif ot_v == 'ts' and mode == 3:
                lambda_mat_col = 1
                folder = 'without Grass transformation using training site'
            elif ot_v == 'cm' and mode == 4:
                lambda_mat_col = 2
                folder = 'with Grass transformation using classification map'
            elif ot_v == 'cm' and mode == 3:
                lambda_mat_col = 3
                folder = 'without Grass transformation using classification map'
            DirPath = '//10.4.0.1/d$/Aria-data/Minervois_2016/' + scene[:6] + scene[-1] + '/' + folder + '/'
            new_folder = 'REG_GS'
            if not os.path.exists(DirPath + new_folder):
                os.makedirs(DirPath + new_folder)
            DirPath = DirPath + new_folder + '/'

            lambda_reg = Lambda_mat[lambda_mat_row][lambda_mat_col]
            # regCL_gs = RegCL_gs_mat[lambda_mat_row][lambda_mat_col]
            regCL_lap = RegCL_lap_mat[lambda_mat_row][lambda_mat_col]
            print('Lambda :', lambda_reg, '; scene: ', scene, '; version: ', ot_v, '; 3 or 4 class:', mode)
            print(DirPath)
            if lambda_reg != 0:
                for base in OT_type:
                    if base == 'EMD':
                        ot_base = ot.da.EMDTransport()
                        ot_base.fit(Xs=Xs, Xt=Xt)
                        transp_Xt_emd = ot_base.inverse_transform(Xt=Xt)
                        #entire_Tdata_trans(scene, base, Data_target, ot_base, lambda_reg, regCL_gs, regCL_lap,
                        #                  ot_v, mode, DirPath)
                    elif base == 'Sinkhorn':
                        # for lambda_reg in range(40000, 1000000, 50000):
                        #    print('lambda : ', lambda_reg)
                        ot_base = ot.da.SinkhornTransport(reg_e=lambda_reg, max_iter=20000, verbose=True)
                        ot_base.fit(Xs=Xs, Xt=Xt)
                        transp_Xt_sinkhorn = ot_base.inverse_transform(Xt=Xt)
                        #entire_Tdata_trans(scene, base, Data_target, ot_base, lambda_reg, regCL_gs, regCL_lap,
                        #                  ot_v, mode, DirPath)
                    elif base == 'L1L2':
                        regCL_gs = 1000
                        while regCL_gs <= 1000000:
                            for lambda_reg in range(40000, 1000000, 200000):
                                ot_base = ot.da.SinkhornL1l2Transport(reg_e=lambda_reg, reg_cl=regCL_gs, max_iter=100,
                                                                      max_inner_iter=20000, verbose=True)
                                ot_base.fit(Xs=Xs, ys=ys, Xt=Xt)
                                transp_Xt_l1l2 = ot_base.inverse_transform(Xt=Xt)
                                entire_Tdata_trans(scene, base, Data_target, ot_base, lambda_reg, regCL_gs, regCL_lap,
                                            ot_v, mode, DirPath)
                            regCL_gs *= 10

                    elif base == 'Laplace':
                        # regCL_lap = 0.01
                        # while regCL_lap <= 1000000:
                        ot_base = ot.da.EMDLaplaceTransport(reg_lap=regCL_lap, reg_src=1, max_iter=200,
                                                            verbose=True,
                                                            max_inner_iter=150000)
                        ot_base.fit(Xs=Xs, ys=ys, Xt=Xt)
                        transp_Xt_emd_laplace = ot_base.inverse_transform(Xt=Xt)
                        #entire_Tdata_trans(scene, base, Data_target, ot_base, lambda_reg, regCL_gs, regCL_lap,
                        #                   ot_v, mode, DirPath)
                        #    regCL_lap *= 10
                # plot_transport(scene, ot_v, mode, DirPath)


'''
# preparing coupling matrix 

pl.figure(1, figsize=(10, 5))
#pl.subplot(1, 3, 2)
coup=ot_sinkhorn.coupling_
max=np.max(coup)
min=np.min(coup)
rag=max-min
a=np.where((min <= coup) & (coup < rag/10))
coup[a]=1
a=np.where((rag/10 <= coup) & (coup < rag/5))
coup[a]=2
a=np.where((rag/5 <= coup) & (coup < rag/2))
coup[a]=3
a=np.where((rag/2 <= coup) & (coup <= max))
coup[a]=4

cmap = ListedColormap(["red", "lightgreen", "darkgreen", "yellow"])
pl.imshow(coup, interpolation='nearest', cmap=cmap)
pl.xticks([])
pl.yticks([])
pl.title('Optimal coupling\nSinkhornTransport')
pl.show()

# since the matrix is so big, the visualization of the matrix is not applicable
'''

