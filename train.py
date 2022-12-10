 #!/usr/bin/env python

import numpy as np
import tensorflow as tf
from tensorflow_graphics.geometry.transformation import euler, rotation_matrix_3d
import pickle
import os
import datetime
from scipy.spatial.transform import Rotation as R
from tensorflow.linalg import inv

from tensorflow import keras
from tensorflow.keras import layers

from positional_encodings.tf_encodings import TFPositionalEncoding1D, TFPositionalEncoding2D, TFPositionalEncoding3D, TFSummer

v = "v80_float16"
#v = ""

c = 299459792

def load_data():
    with open("training_data/train_data_00" + v + ".txt", "rb") as fp:
        train_data0 = pickle.load(fp)
    with open("training_data/truth_data_00" + v + ".txt", "rb") as fp:
        truth_data0 = pickle.load(fp)

    with open("training_data/train_data_01" + v + ".txt", "rb") as fp:
        train_data1 = pickle.load(fp)
    with open("training_data/truth_data_01" + v + ".txt", "rb") as fp:
        truth_data1 = pickle.load(fp)

    with open("training_data/train_data_02" + v + ".txt", "rb") as fp:
        train_data2 = pickle.load(fp)
    with open("training_data/truth_data_02" + v + ".txt", "rb") as fp:
        truth_data2 = pickle.load(fp)

    with open("training_data/train_data_03" + v + ".txt", "rb") as fp:
        train_data3 = pickle.load(fp)
    with open("training_data/truth_data_03" + v + ".txt", "rb") as fp:
        truth_data3 = pickle.load(fp)


    #x_train_full = np.transpose(np.vstack((train_data0, train_data1, train_data2, train_data3)), (0,2,1)) #(0,1,2) keeps it the same actually
    #y_train_full = np.vstack((truth_data0, truth_data1, truth_data2, truth_data3))
    
    x_train_full = np.transpose(train_data0, (0,2,1)) #(0,1,2) keeps it the same actually
    y_train_full = truth_data0
    return x_train_full, y_train_full

#def load_data():
#    with open("training_data/train_data_00" + v + ".txt", "rb") as fp:
#        train_data0 = pickle.load(fp)
#    with open("training_data/truth_data_00" + v + ".txt", "rb") as fp:
#        truth_data0 = pickle.load(fp)
#
#    with open("training_data/train_data_01" + v + ".txt", "rb") as fp:
#        train_data1 = pickle.load(fp)
#    with open("training_data/truth_data_01" + v + ".txt", "rb") as fp:
#        truth_data1 = pickle.load(fp)
#
#    with open("training_data/train_data_02" + v + ".txt", "rb") as fp:
#        train_data2 = pickle.load(fp)
#    with open("training_data/truth_data_02" + v + ".txt", "rb") as fp:
#        truth_data2 = pickle.load(fp)
#
#    with open("training_data/train_data_03" + v + ".txt", "rb") as fp:
#        train_data3 = pickle.load(fp)
#    with open("training_data/truth_data_03" + v + ".txt", "rb") as fp:
#        truth_data3 = pickle.load(fp)
#
#    with open("training_data/train_data_04" + v + ".txt", "rb") as fp:
#        train_data4 = pickle.load(fp)
#    with open("training_data/truth_data_04" + v + ".txt", "rb") as fp:
#        truth_data4 = pickle.load(fp)
#
#    with open("training_data/train_data_05" + v + ".txt", "rb") as fp:
#        train_data5 = pickle.load(fp)
#    with open("training_data/truth_data_05" + v + ".txt", "rb") as fp:
#        truth_data5 = pickle.load(fp)
#
#    with open("training_data/train_data_06" + v + ".txt", "rb") as fp:
#        train_data6 = pickle.load(fp)
#    with open("training_data/truth_data_06" + v + ".txt", "rb") as fp:
#        truth_data6 = pickle.load(fp)
#
#    with open("training_data/train_data_07" + v + ".txt", "rb") as fp:
#        train_data7 = pickle.load(fp)
#    with open("training_data/truth_data_07" + v + ".txt", "rb") as fp:
#        truth_data7 = pickle.load(fp)
#
#
#    x_train_full = np.transpose(np.vstack((train_data0, train_data1, train_data2, train_data3, train_data4, train_data5, train_data6, train_data7)), (0,2,1)) #(0,1,2) keeps it the same actually
#    y_train_full = np.vstack((truth_data0, truth_data1, truth_data2, truth_data3, truth_data4, truth_data5, truth_data6, truth_data7))
#    return x_train_full, y_train_full

def err(y_pred, y):
    #Model Output (in velo coord frame)
    eul_out = y_pred[:, :3]
    t_out = tf.reshape(y_pred[:, 3:], [-1, 3, 1])
    
    #Ground Truth Info (in cam coord frame)
    # old: [0,3,1,2], new: [0,1,2,3]
    Rt2 = tf.reshape(y[:, 0, :], [-1, 3, 4])
    Rt3 = tf.reshape(y[:, 3, :], [-1, 3, 4])
    P0 = tf.reshape(y[:, 1, :], [-1, 3, 4]) #Not needed
    T = tf.reshape(y[:, 2, :], [-1, 3, 4]) #velo -> left camera coord frame
    Tr = T[:, :, :3]
    Tt = tf.reshape(T[:, :, 3], [-1, 3, 1])
    
    t2c = tf.reshape(Rt2[:, :, 3], [-1, 3, 1])
    t3c = tf.reshape(Rt3[:, :, 3], [-1, 3, 1])
    R2c = Rt2[:, :, :3]
    R3c = Rt3[:, :, :3]
    
    tv_truth = inv(Tr) @ (R3c @ inv(R2c) @ Tt + (t3c - R3c @ inv(R2c) @ t2c) - Tt)
    Rv_truth = inv(Tr) @ R3c @ inv(R2c) @ Tr
    
    eul_truth_velo = euler.from_rotation_matrix(Rv_truth)
    
    rand = np.random.random()
    if rand < 0.01:
        print("test, eul truth:", eul_truth_velo[0,:])
        print("test, eul out:", eul_out[0,:])

    loss1 = tf.math.reduce_euclidean_norm(eul_truth_velo - eul_out, 1)
    loss2 = tf.math.reduce_euclidean_norm(tv_truth - t_out, 1)
    return tf.reduce_sum(loss1), tf.reduce_sum(loss2)


def eul_loss(eul_out, y):
    #Model Output (in velo coord frame)
    #eul_out = y_pred[:, :3]

    #Ground Truth Info (in cam coord frame)
    
    # old: [0,3,1,2], new: [0,1,2,3]
    Rt2 = tf.reshape(y[:, 0, :], [-1, 3, 4])
    Rt3 = tf.reshape(y[:, 3, :], [-1, 3, 4])
    P0 = tf.reshape(y[:, 1, :], [-1, 3, 4]) #Not needed
    T = tf.reshape(y[:, 2, :], [-1, 3, 4]) #velo -> left camera coord frame
    Tr = T[:, :, :3]
    
    R2c = Rt2[:, :, :3]
    R3c = Rt3[:, :, :3]
    
    Rv_truth = inv(Tr) @ R3c @ inv(R2c) @ Tr
    eul_truth_velo = euler.from_rotation_matrix(Rv_truth)#*180/np.pi

    loss = keras.losses.MeanSquaredError()(eul_truth_velo, eul_out)
    return loss
 


def trans_loss(y_pred, y):
    #Model Output (in velo coord frame)
    t_out = tf.reshape(y_pred, [-1, 3, 1])

    #Ground Truth Info (in cam coord frame)
    # old: [0,3,1,2], new: [0,1,2,3]
    Rt2 = tf.reshape(y[:, 0, :], [-1, 3, 4])
    Rt3 = tf.reshape(y[:, 3, :], [-1, 3, 4])
    P0 = tf.reshape(y[:, 1, :], [-1, 3, 4]) #Not needed
    T = tf.reshape(y[:, 2, :], [-1, 3, 4]) #velo -> left camera coord frame
    Tr = T[:, :, :3]
    Tt = tf.reshape(T[:, :, 3], [-1, 3, 1])
    
    t2c = tf.reshape(Rt2[:, :, 3], [-1, 3, 1])
    t3c = tf.reshape(Rt3[:, :, 3], [-1, 3, 1])
    R2c = Rt2[:, :, :3]
    R3c = Rt3[:, :, :3]
    
    tv_truth = inv(Tr) @ (R3c @ inv(R2c) @ Tt + (t3c - R3c @ inv(R2c) @ t2c) - Tt)
    
    loss = keras.losses.MeanSquaredError()(tv_truth, t_out)
    return loss
 


def transformer_encoder(input, head_size, num_heads, ff_dim, dropout=0):

    #Self-Attention
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(input, input)
    x = tf.keras.layers.Add()([x, input])
    res1 = layers.LayerNormalization(epsilon=1e-6)(x)
    
    #Feed Forward
    x = res1
    for dim in ff_dim:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
    
    x = layers.Dense(input.shape[-1], activation="relu")(x)
    #x = layers.Dropout(dropout)(x)
    x = tf.keras.layers.Add()([x, res1])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x
    
    
def transformer_decoder(input1, input2, head_size, num_heads, ff_dim, dropout=0):

    # Self-Attention
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(input2, input2, use_causal_mask=True)
    x = tf.keras.layers.Add()([x, input2])
    res1 = layers.LayerNormalization(epsilon=1e-6)(x)
    
    #Cross-Attention
    x, att_scores = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(res1, input1, return_attention_scores=True)
    x = tf.keras.layers.Add()([x, res1])
    res2 = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Feed Forward
    x = res2
    for dim in ff_dim:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
    
    x = layers.Dense(input1.shape[-1], activation="relu")(x)
    #x = layers.Dropout(dropout)(x)
    x = tf.keras.layers.Add()([x, res2])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x, att_scores
    

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_encoder_blocks,
    num_decoder_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
    shapes=[],
    mlp_activation="relu",
    div=[128]
):
    inputs = keras.Input(shape=input_shape)
    m1 = inputs[:, :, :3]
    m2 = inputs[:, :, 3:]
    
    for shape in shapes:
        m1 = tf.reshape(m1, [-1, shape[0], shape[1]])
        m2 = tf.reshape(m2, [-1, shape[0], shape[1]])
        
        for d in div:
            m1 = layers.Dense(d, activation="tanh")(m1)
            m2 = layers.Dense(d, activation="tanh")(m2)

        #m1 = TFSummer(TFPositionalEncoding1D(d))(m1)
        #m2 = TFSummer(TFPositionalEncoding1D(d))(m2)
        m1 = TFSummer(TFPositionalEncoding1D(shape[1]))(m1)
        m2 = TFSummer(TFPositionalEncoding1D(shape[1]))(m2)
        
        for _ in range(num_encoder_blocks):
            m1 = transformer_encoder(m1, head_size, num_heads, ff_dim, dropout)
        for _ in range(num_decoder_blocks):
            m2, att_scores = transformer_decoder(m1, m2, head_size, num_heads, ff_dim, dropout)
    
    
    x = m2
    x = layers.Flatten()(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation=mlp_activation)(x)
        x = layers.Dropout(mlp_dropout)(x)
        
    output = layers.Dense(3)(x)
    return keras.Model(inputs, output)




def build_and_train(x_train_full, y_train_full):
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_e_loss = tf.keras.metrics.Mean(name='train_e_loss')
    train_t_loss = tf.keras.metrics.Mean(name='train_t_loss')
    test_err_eul = tf.keras.metrics.Mean(name='test_err_eul')
    test_err_trans = tf.keras.metrics.Mean(name='test_err_trans')
    val_err_eul = tf.keras.metrics.Mean(name='val_err_eul')
    val_err_trans = tf.keras.metrics.Mean(name='val_err_trans')
    
    #@tf.function
    
    train_loss_all = []
    test_loss_all = []
    #@tf.function
    def train():
        count = 0
        for x, y in train_dataset:
            with tf.GradientTape() as tape1:
                t_pred = t_model(x)
                curr_t_loss = trans_loss(t_pred, y)

            grad1 = tape1.gradient(curr_t_loss, t_model.trainable_variables)
            t_optimizer.apply_gradients(zip(grad1, t_model.trainable_variables))
            train_t_loss(curr_t_loss)
            
            with tf.GradientTape() as tape2:
                e_pred = e_model(x)
                curr_e_loss = eul_loss(e_pred, y)

            grad2 = tape2.gradient(curr_e_loss, e_model.trainable_variables)
            e_optimizer.apply_gradients(zip(grad2, e_model.trainable_variables))
            train_e_loss(curr_e_loss)
            train_loss_all.append((curr_e_loss.numpy(), curr_t_loss.numpy()))
            
            print(100*count/len(train_dataset), curr_e_loss.numpy(), curr_t_loss.numpy())
            count += 1
        
        # check acc against training data
        for x_test, y_test in train_dataset:
            t_pred = t_model(x_test)
            e_pred = e_model(x_test)
            y_pred = tf.concat([e_pred, t_pred], 1)
            t_err_eul, t_err_trans = err(y_pred, y_test)
            test_err_eul(t_err_eul)
            test_err_trans(t_err_trans)
            test_loss_all.append((t_err_eul.numpy(), t_err_trans.numpy()))
            
        # check acc against test data
        for x_val, y_val in test_dataset:
            t_pred = t_model(x_val)
            e_pred = e_model(x_val)
            y_pred = tf.concat([e_pred, t_pred], 1)
            v_err_eul, v_err_trans = err(y_pred, y_val)
            val_err_eul(v_err_eul)
            val_err_trans(v_err_trans)

        
         
    # Models
    input_shape = ((x_train_full.shape[1], 6))
    
    t_model = build_model(
        input_shape,
        head_size=120,#50 -> d_model/num_heads
        num_heads=8,#5
        ff_dim=[],#750 (125 good too)
        num_encoder_blocks=1,#3
        num_decoder_blocks=1,
        mlp_units=[],#, 3, 27, 3, 27],#[27]
        mlp_dropout=0.1,#0.1
        dropout=0.2,#0.1
        shapes=[(250, 960)],
        mlp_activation="relu",
        div=[]
    )
    
    e_model = build_model(
        input_shape,
        head_size=120,#50 -> d_model/num_heads
        num_heads=8,#5
        ff_dim=[],#750 (125 good too)
        num_encoder_blocks=1,#3
        num_decoder_blocks=1,
        mlp_units=[],#, 3, 27, 3, 27],#[27]
        mlp_dropout=0.1,#0.1
        dropout=0.2,#0.1
        shapes=[(250, 960)],
        mlp_activation="relu",
        div=[]
    )
    
    
    #e_model.load_weights('./trained_weights/e_v0_run0')
    #t_model.load_weights('./trained_weights/t_v0_run0')
    
    
    e_model.summary()
    t_model.summary()
    #################################################




    # Hyper-Parameters
    #################################################
    weights_to_load = 0
    epochs = 30
    batch_size = 4
    #train_set_size = 100
    
    #################################################
    boundaries1 = [1000, 1000, 1000]
    values1 = [0.00005, 0.000005, 0.0000025, 0.000001]
    learning_rate_fn1 = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries1, values1)
    
    boundaries2 = [1000, 1000, 1000]
    values2 = [0.00005, 0.000005, 0.0000025, 0.000001]
    learning_rate_fn2 = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries2, values2)
    
    t_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn1)
    e_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn2)
    
    #################################################
    # Build Dataset
    #################################################

    test_idx = round(0.95*x_train_full.shape[0])
    x_train = x_train_full[:test_idx, :, :]
    y_train = y_train_full[:test_idx, :, :]

    x_test = x_train_full[test_idx:, :, :]
    y_test = y_train_full[test_idx:, :, :]

#    x_train = x_train_full
#    y_train = y_train_full
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        
    print("Beginning Dataset consruction...")
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(13000).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(3000).batch(batch_size)
    print("Dataset consruction completed!")
        
    # Mutiple Training Loops....
    #################################################
    
    for run in range(10):
        
        #################################################
        for epoch in range(epochs):
            train_e_loss.reset_states()
            train_t_loss.reset_states()
            test_err_eul.reset_states()
            test_err_trans.reset_states()
            val_err_eul.reset_states()
            val_err_trans.reset_states()
            
            train()
            
            template = '\nRun {}, Epoch {}, Eul Loss: {}, Trans Loss: {}, \nEul Test Err: {}, Trans Test Err: {}, \nEul Val Err: {}, Trans Val Err: {}\n'
            
            print(template.format(run + 1, epoch + 1, train_e_loss.result(), train_t_loss.result(), test_err_eul.result(), test_err_trans.result(), val_err_eul.result(), val_err_trans.result()))
            

        
        e_model.save_weights('./trained_weights/e_v80_run' + str(weights_to_load))
        t_model.save_weights('./trained_weights/t_v80_run' + str(weights_to_load))
        print('Saving weights to e_v80_run' + str(weights_to_load))
        print('Saving weights to t_v80_run' + str(weights_to_load))
        weights_to_load += 1
        with open("train_loss_all_v80" + str(weights_to_load), "wb") as fp:
            pickle.dump(train_loss_all, fp)
        with open("test_loss_all_v80" + str(weights_to_load), "wb") as fp:
            pickle.dump(test_loss_all, fp)
        

        
    #model.summary()
    #################################################

