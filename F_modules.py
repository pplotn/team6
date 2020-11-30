print('Modules import started')
import glob
from itertools import islice, tee
# import m8r as sf
from collections import Counter
import datetime
import math
import fnmatch
import skimage
from skimage.transform import resize as imresize
from skimage.transform import resize as resize
import scipy
import random
import numpy as np
import matplotlib
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io
from matplotlib.transforms import Bbox, TransformedBbox, Affine2D
from matplotlib import  tight_bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.matlib as npm
# import pandas as pd
import itertools
from itertools import islice
# import more_itertools
from collections import deque
import h5py
from scipy import signal
from scipy.ndimage.filters import gaussian_filter,gaussian_filter1d
from scipy.ndimage.interpolation import map_coordinates
from scipy.fftpack import fft2, ifft2
from collections import namedtuple
import random
import os, sys
from shutil import copyfile
import time
import subprocess
import sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# from sklearn.externals import joblib
from functools import reduce


import tensorflow as tf
print('TensorFlow\t{}'.format(tf.__version__))
a=(tf.__version__)
ind=a.split('.')
ver=float(ind[0]+'.'+ind[1])
if ver==1.12:
    ###########################   TF 1.13 and keras
    import tensorflow as tf
    import keras
    print('Keras\t\t{}'.format(keras.__version__))
    print('TensorFlow\t{}'.format(tf.__version__))
    from keras.callbacks import *
    from keras.optimizers import *
    from keras.backend.tensorflow_backend import set_session
    from keras.layers import Conv1D,Conv2D, MaxPooling1D,MaxPooling2D, Dense, Dropout, Reshape, Flatten, BatchNormalization
    from keras.models import Model, Sequential, load_model
    from keras import backend as K
    from keras.utils import plot_model, multi_gpu_model,Sequence
    # from keras.layers import Dense, Dropout, Reshape, BatchNormalization, GlobalAveragePooling2D, UpSampling1D, UpSampling2D
    from keras.layers import *
    # TensorFlow wizardry
    config = tf.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    # Create a session with the above options specified.
    K.tensorflow_backend.set_session(tf.Session(config=config))
else:
    ###########################   TF 2.0
    from tensorflow.keras.models import *
    from tensorflow.keras.layers import *
    from tensorflow.keras.optimizers import *
    from tensorflow.keras.callbacks import *
    from tensorflow.keras import regularizers
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.utils import multi_gpu_model,Sequence
    
    # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # tf.compat.v1.disable_eager_execution()
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
###########################
import logging
print('Functions import started')
## modules for distorting velocity models
from numpy.random import seed, randint
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage, misc
from scipy.ndimage.interpolation import map_coordinates
import multiprocessing
