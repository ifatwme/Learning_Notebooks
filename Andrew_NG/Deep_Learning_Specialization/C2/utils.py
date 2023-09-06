import numpy as np
import matplotlib.pyplot as plt
import zipfile
from pathlib import Path
import os

import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable

# Load a dataset from kaggle given the API command
def load_dataset_kaggle(kaggle):
    dataset_name = kaggle.split('/')[-1]
    print('[NOTE] Start loading', dataset_name)
    data_path = Path('./dataset/'+dataset_name+'/')
    if data_path.is_dir():
        print(f"[NOTE] {data_path} directory exists.")
    else:
        print(f"[NOTE] Did not find {data_path} directory, creating one...")
        data_path.mkdir()

    print(f'[NOTE] Start downloading dataset from kaggle.')
    os.system(kaggle)

    with zipfile.ZipFile(dataset_name+'.zip', 'r') as zf:
        print('[NOTE] Unzipping the dataset')
        zf.extractall(data_path)
        zf.close()

# Inspect give directory
def walk_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print('There are {} directories & {} files in {}'
              .format(len(dirnames), len(filenames), dirpath))

def linear_function_test(target):
    result = target()
    print(result)

    assert type(result) == EagerTensor, "Use the TensorFlow API"
    assert np.allclose(result, [[-2.15657382], [ 2.95891446], [-1.08926781], [-0.84538042]]), "Error"
    print("\033[92mAll test passed")


def sigmoid_test(target):
    result = target(0)
    
    print ("type: " + str(type(result)))
    print ("dtype: " + str(result.dtype))
    print ("sigmoid(-1) = " + str(result))
    print ("sigmoid(0) = " + str(target(0.0)))
    print ("sigmoid(12) = " + str(target(12)))

    assert(type(result) == EagerTensor)
    assert(result.dtype == tf.float32)
    assert target(0) == 0.5, "Error"
    assert target(-1) == 0.26894143, "Error"
    assert target(12) == 0.99999386, "Error"

    print("\033[92mAll test passed")

def one_hot_matrix_test(target):
    label = tf.constant(1)
    depth = 4
    result = target(label, depth)
    print("Test 1:",result)
    assert result.shape[0] == depth, "Use the parameter depth"
    assert np.allclose(result, [0., 1. ,0., 0.] ), "Wrong output. Use tf.one_hot"

    label_2 = [2]
    result = target(label_2, depth)
    print("Test 2:", result)
    assert result.shape[0] == depth, "Use the parameter depth"
    assert np.allclose(result, [0., 0. ,1., 0.] ), "Wrong output. Use tf.reshape as instructed"
    
    print("\033[92mAll test passed")


def initialize_parameters_test(target, dims):
    parameters = target(dims)

    values = {"W1": (25, 12288),
              "b1": (25, 1),
              "W2": (12, 25),
              "b2": (12, 1),
              "W3": (10, 12),
              "b3": (10, 1)}

    for key in parameters:
        print(f"{key} shape: {tuple(parameters[key].shape)}")
        assert type(parameters[key]) == ResourceVariable, "All parameter must be created using tf.Variable"
        assert tuple(parameters[key].shape) == values[key], f"{key}: wrong shape"
        assert np.abs(np.mean(parameters[key].numpy())) < 0.5,  f"{key}: Use the GlorotNormal initializer"
        assert np.std(parameters[key].numpy()) > 0 and np.std(parameters[key].numpy()) < 1, f"{key}: Use the GlorotNormal initializer"

    print("\033[92mAll test passed")

def forward_propagation_test(target, examples, params):
    minibatches = examples.batch(2)
    for minibatch in minibatches:
        forward_pass = target(tf.transpose(minibatch), params)
        print(forward_pass)
        assert type(forward_pass) == EagerTensor, "Your output is not a tensor"
        assert forward_pass.shape == (10, 2), "Last layer must use W3 and b3"
        break
    

    print("\033[92mAll test passed")

def compute_cost_test(target, Y):
    pred = tf.constant([[ 2.4048107,   5.0334096 ],
             [-0.7921977,  -4.1523376 ],
             [ 0.9447198,  -0.46802214],
             [ 1.158121,    3.9810789 ],
             [ 4.768706,    2.3220146 ],
             [ 6.1481323,   3.909829  ],
             [ 0.9447198,  -0.46802214],
             [ 1.158121,    3.9810789 ],
             [ 4.768706,    2.3220146 ],
             [ 6.1481323,   3.909829  ]])
    minibatches = Y.batch(2)
    for minibatch in minibatches:
        result = target(pred, tf.transpose(minibatch))
        break
        
    print(result)
    assert(type(result) == EagerTensor), "Use the TensorFlow API"
    print (np.abs(result - (0.25361037 + 0.5566767) / 2.0) ), "Test does not match. Did you get the mean of your cost functions?"

    print("\033[92mAll test passed")
