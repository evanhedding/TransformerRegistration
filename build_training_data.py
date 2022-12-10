#!/usr/bin/env python

import numpy as np
import os
import pickle

max_seq = 0
for seq in range(0,4):
    cloud_dir = 'dataset/sequences/0' + str(seq) + '/velodyne/'
    truth_dir = 'dataset/ground_truth/0' + str(seq) + '.txt'
    calib_dir = 'dataset/calibr/0' + str(seq) + '/calib.txt'

    data_start = 1
    data_end = 301

    calib_dict = {}
    with open(calib_dir) as f:
        for line in f:
            (key, val) = line.split(": ", 1)
            list = val.split()
            for i in range(len(list)):
                try: list[i] = np.float32(float(list[i]))
                except: None
            calib_dict[key] = np.asarray(list)
    
    truth_list = []
    with open(truth_dir) as f:
        for line in f:
            list = line.split()
            for i in range(len(list)):
                try: list[i] = np.float32(float(list[i]))
                except: None
            
            arr = np.array([np.asarray(list), calib_dict['P0'], calib_dict['Tr']])
            #[(ground truth), (P0), (Tr)]
            truth_list.append(arr)
 
            
    truth_data = []
    train_data = []
    PC_size = 80000
    prev1_cloud = np.zeros((PC_size, 4))
#    prev2_cloud = np.zeros((PC_size, 4))
#    prev3_cloud = np.zeros((PC_size, 4))
    col_size = 3 # 3 = w/o reflectivity, 4 = w/ reflectivity

    print("Loading Point Cloud data " + str(seq) + "..." )
    for i, filename in enumerate(sorted(os.scandir(cloud_dir), key=lambda e: e.name)):
        if filename.is_file():
            curr_cloud = np.fromfile(filename, '<f4').reshape((-1, 4))
            if curr_cloud.shape[0] > max_seq: max_seq = curr_cloud.shape[0]
            rand_set1 = np.sort(np.random.choice(curr_cloud.shape[0], PC_size, replace=False))
            curr_cloud = curr_cloud[rand_set1, :]
            
            if i >= 1:
                combined  = np.hstack((prev1_cloud[:,:col_size], curr_cloud[:,:col_size]))
                train_data.append(combined.T)
                truth_data.append(np.vstack((truth_list[i - 1], truth_list[i])))
#            if i >= 2:
#                combined  = np.hstack((prev2_cloud[:,:col_size], curr_cloud[:,:col_size]))
#                train_data.append(combined.T)
#                truth_data.append(np.vstack((truth_list[i - 2], truth_list[i])))
#            if i >= 3:
#                combined  = np.hstack((prev3_cloud[:,:col_size], curr_cloud[:,:col_size]))
#                train_data.append(combined.T)
#                truth_data.append(np.vstack((truth_list[i - 3], truth_list[i])))

            prev1_cloud = curr_cloud
#            prev2_cloud = prev1_cloud
#            prev3_cloud = prev2_cloud

#
#    print(truth_data[0])
#    print(truth_data[1])
#    print(truth_data[2])

    print("max_seq: ", max_seq)
    print("Point Cloud data created! Saving to pickle...")
    
    
    #rand_set = np.random.choice(len(train_data), data_end - 1, replace=False)
    train_data = np.asarray(train_data, dtype=np.float16)#[rand_set, :, :]
    truth_data = np.asarray(truth_data)#[rand_set, :, :]
    print(truth_data.shape, train_data.shape)

    with open("graphics/training_data/train_data_0" + str(seq) + "v80_float16.txt", "wb") as fp:
        pickle.dump(train_data, fp)
        
    with open("graphics/training_data/truth_data_0" + str(seq) + "v80_float16.txt", "wb") as fp:
        pickle.dump(truth_data, fp)
