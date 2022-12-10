#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from tensorflow_graphics.geometry.transformation import euler, rotation_matrix_3d
import pickle
import os
import datetime
from scipy.spatial.transform import Rotation as R
from tensorflow.linalg import inv
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from new_train import *

# Models
input_shape = ((40000, 6))

t_model = build_model(
    input_shape,
    head_size=100,#50 -> d_model/num_heads
    num_heads=5,#5
    ff_dim=[1000],#750 (125 good too)
    num_encoder_blocks=8,#3
    num_decoder_blocks=8,
    mlp_units=[],#, 3, 27, 3, 27],#[27]
    mlp_dropout=0.2,#0.1
    dropout=0.2,#0.1
    shapes=[(20, 6000)],
    mlp_activation="relu"
)

e_model = build_model(
    input_shape,
    head_size=100,#50 -> d_model/num_heads
    num_heads=5,#5
    ff_dim=[1000],#750 (125 good too)
    num_encoder_blocks=8,#3
    num_decoder_blocks=8,
    mlp_units=[],#, 3, 27, 3, 27],#[27]
    mlp_dropout=0.2,#0.1
    dropout=0.2,#0.1
    shapes=[(20, 6000)],
    mlp_activation="relu"
)

#e_model.load_weights('./trained_weights/e_v8_run9')
#t_model.load_weights('./trained_weights/t_v8_run9')


def load_data():
    with open("training_data/train_data_03v08.txt", "rb") as fp:
        train_data0 = pickle.load(fp)
    with open("training_data/truth_data_03v08.txt", "rb") as fp:
        truth_data0 = pickle.load(fp)
        
    x_train_full = np.transpose(train_data0, (0,2,1)) #(0,1,2) keeps it the same actually
    y_train_full = truth_data0
    return x_train_full, y_train_full
 


# Make arrays of euler angles, tranlsation vectors vs. timesteps (Point Cloud series)
seq_myts = []
seq_gts = []
e_truth = []
e_my = []
t_truth = []
t_my = []

fig = plt.figure(1)
ax = fig.add_subplot()

gt = np.ones((3,1))
my_t = np.ones((3,1))
init = np.ones((3,1))

x_train_full, y_train_full = load_data()

for i in range(len(x_train_full)):
    Rt2 = tf.reshape(y_train_full[i, 0, :], [3, 4])
    Rt3 = tf.reshape(y_train_full[i, 3, :], [3, 4])
    P0 = tf.reshape(y_train_full[i, 1, :], [3, 4]) #Not needed
    T = tf.reshape(y_train_full[i, 2, :], [3, 4]) #velo -> left camera coord frame
    Tr = T[:, :3]
    Tt = tf.reshape(T[:, 3], [3, 1])
    
    t2c = tf.reshape(Rt2[:, 3], [3, 1])
    t3c = tf.reshape(Rt3[:, 3], [3, 1])
    R2c = Rt2[:, :3]
    R3c = Rt3[:, :3]
    
    R = inv(Tr) @ R3c @ inv(R2c) @ Tr
    t = inv(Tr) @ (R3c @ inv(R2c) @ Tt + (t3c - R3c @ inv(R2c) @ t2c) - Tt)
    gt = R @ gt + t
    gt = R3c @ init + t3c #cam frame
    
    input = np.reshape(x_train_full[i], (1, 40000, 6))

    eul_out = e_model(input).numpy()
    t_out = t_model(input).numpy()
    t_out = t_out.reshape((3,1))
    eul_out = eul_out.reshape((3,))

#    print("")
#    print('tout', t_out)
#    print("ttruth", t)
#    print("eul_out", eul_out)
#    print("eultruth", euler.from_rotation_matrix(R))
#    print("")

    rot = rotation_matrix_3d.from_euler(eul_out)
    my_t = rot @ my_t + t_out
    
    print("my_t:  ", my_t)
    print("gt:  ", gt)
    seq_gts.append(gt)
    seq_myts.append(my_t)
    e_truth.append(euler.from_rotation_matrix(R))
    e_my.append(eul_out)
    t_truth.append(t)
    t_my.append(t_out)

#deg = np.asarray(deg)

#
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.scatter(trans[:,0], trans[:,1], trans[:,2])

seq_gts = np.asarray(seq_gts)
seq_myts = np.asarray(seq_myts)
t_truth = np.asarray(t_truth)
e_truth = np.asarray(e_truth)
t_my = np.asarray(t_my)
e_my = np.asarray(e_my)

fig1 = plt.figure(1)
plt.plot(seq_gts[:,1], seq_gts[:,0])
plt.plot(seq_myts[:,1], seq_myts[:,0])


fig2, axs2 = plt.subplots(3)
fig2.suptitle('Translation Vector, Ground Truth vs. Output')
axs2[0].plot(t_truth[:,0])
axs2[0].plot(t_my[:,0])
axs2[1].plot(t_truth[:,1])
axs2[1].plot(t_my[:,1])
axs2[2].plot(t_truth[:,2])
axs2[2].plot(t_my[:,2])

fig3, axs3 = plt.subplots(3)
fig3.suptitle('Euler Angles (Rad), Ground Truth vs. Output')
axs3[0].plot(e_truth[:,0])
axs3[0].plot(e_my[:,0])
axs3[1].plot(e_truth[:,1])
axs3[1].plot(e_my[:,1])
axs3[2].plot(e_truth[:,2])
axs3[2].plot(e_my[:,2])

plt.show()
