#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from tensorflow_graphics.geometry.transformation import euler
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d
import pickle
import os
from train import *

x_train_full, y_train_full = load_data()
build_and_train(x_train_full, y_train_full)
