import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

from qiskit.tools.monitor import job_monitor
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit import Aer, execute, IBMQ
from qiskit.extensions import XGate, UnitaryGate

import sys
import os
import time
import functools
import pandas as pd

from lib.utils import *
from lib.U_layer import *
from lib.P_layer import *

interest_num = [3,6]
ori_img_size = 28
img_size = 4
# number of subprocesses to use for data loading
num_workers = 12
# how many samples per batch to load
batch_size = 1
inference_batch_size = 1
data_path = './data'

qc_shots = 8192

# convert data to torch.FloatTensor
transform = transforms.Compose([transforms.Resize((ori_img_size,ori_img_size)),
                                transforms.ToTensor()])
# Path to MNIST Dataset
train_data = datasets.MNIST(root=data_path, train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root=data_path, train=False,
                                  download=True, transform=transform)

train_data = select_num(train_data,interest_num)
test_data =  select_num(test_data,interest_num)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    num_workers=num_workers, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=inference_batch_size, 
    num_workers=num_workers, shuffle=True, drop_last=True)


# Model initialization
weight_1_1 = torch.tensor([1.,  1.,  1.,  1.,  1.,  1.,  1., -1.,  1., -1.,  1., -1.,  1.,  1.,    1.,  1.])
weight_1_2 = torch.tensor([-1., -1., -1., -1., -1., -1., -1., -1., -1.,  1., -1.,  1., -1., -1.,-1., -1.])

weight_2_1 = torch.tensor([1.,  -1.])
norm_flag_1 = True
norm_para_1 = torch.tensor(0.3060)

weight_2_2 = torch.tensor([-1.,  -1.])
norm_flag_2 = False
norm_para_2 = torch.tensor(0.6940)

weights = [weight_1_1, weight_1_2, weight_2_1, weight_2_2]
flags = [norm_flag_1, norm_flag_2]
params = [norm_para_1, norm_para_2]

saving = True
quiet = True
simulation = True

p_pred = []
u_pred = []
p_prob = []
u_prob = []
itr = 0

for batch_idx, (data, target) in enumerate(test_loader):

    torch.set_printoptions(threshold=sys.maxsize)
    # print("Batch Id: {}, Target: {}".format(batch_idx,target))
    quantum_matrix, qantum_data = data_pre_pro(
        torchvision.utils.make_grid(data), img_size, verbose=False)

    p_circ = p_circ_gen(quantum_matrix, weights, flags, params)
    u_circ = u_circ_gen(quantum_matrix, weights, flags, params)

    # p_circ
    counts = fire_ibmq(p_circ, qc_shots, Simulation=simulation, backend_name='ibmq_santiago', quiet=quiet)
    (mycount, bits) = analyze(counts)
    class_prob = []
    for b in range(bits):
        class_prob.append(float(mycount[b])/qc_shots)
    
    p_prob.append(class_prob)
    result = abs((class_prob.index(max(class_prob)) - target[0]).numpy()) # 0 if correct, 1 if not
    p_pred.append(result)

    if quiet==False:
        print("="*10, "Non-Optimized Circuit", "="*10)
        print("Non-Optimized Circuit Depth:", p_circ.depth())
        print("Result of non-optimized QC:", class_prob)
        print("Prediction class: {}".format(class_prob.index(max(class_prob))))
        print("Target class: {}".format(target[0]))

        if class_prob.index(max(class_prob)) == target[0]:
            print("Correct prediction")
        else:
            print("Incorrect prediction")

        print("="*30)

    # u_circ
    opt_counts = fire_ibmq(u_circ, qc_shots, Simulation=simulation, quiet=quiet)
    (opt_mycount, bits) = analyze(opt_counts)
    opt_class_prob = []
    for b in range(bits):
        opt_class_prob.append(float(opt_mycount[b])/qc_shots)

    u_prob.append(opt_class_prob)
    result = abs((opt_class_prob.index(max(opt_class_prob)) - target[0]).numpy())
    u_pred.append(result)

    if quiet==False:
        print("="*10, "Optimized Circuit", "="*10)
        print("Optimized Circuit Depth:", u_circ.depth())
        print("Result of optimized QC:", opt_class_prob)
        print("Prediction class: {}".format(opt_class_prob.index(max(opt_class_prob))))
        print("Target class: {}".format(target[0]))

        if opt_class_prob.index(max(opt_class_prob)) == target[0]:
            print("Correct prediction")
        else:
            print("Incorrect prediction")

        print("="*30)
    
    itr += 1
    if itr % 100 == 0:
        print("iteration "+itr)
    
    if itr >= 2:
        break

if saving:
    t = time.localtime()
    dir = "./results/"+time.strftime('%m-%d_%H-%M', t)

    if not os.path.exists(dir):
        os.makedirs(dir)

    df_pred = pd.DataFrame([p_pred, u_pred])
    df_pred.to_csv(dir+"/predictions.csv", index=False)

    dfp = pd.DataFrame(p_prob, columns=["p[0]", "p[1]"])
    dfu = pd.DataFrame(u_prob, columns=["u[0]", "u[1]"])

    df_prob = pd.concat([dfp, dfu], axis=1)
    df_pred.to_csv(dir+"/probilities.csv", index=False)