import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from qiskit.tools.monitor import job_monitor
from qiskit import Aer, execute, IBMQ
import qiskit as q


def modify_target(target, interest_num):
    """
    modify the target classes starting from 0. Say, [3,6] -> [0,1]
    """
    for j in range(len(target)):
        for idx in range(len(interest_num)):
            if target[j] == interest_num[idx]:
                target[j] = idx
                break
    new_target = torch.zeros(target.shape[0], 2)
    for i in range(target.shape[0]):
        if target[i].item() == 0:
            new_target[i] = torch.tensor([1, 0]).clone()
        else:
            new_target[i] = torch.tensor([0, 1]).clone()

    return target, new_target


def select_num(dataset, interest_num):
    """
    select sub-set from MNIST
    """
    labels = dataset.targets  # get labels
    labels = labels.numpy()
    idx = {}
    for num in interest_num:
        idx[num] = np.where(labels == num)
    fin_idx = idx[interest_num[0]]
    for i in range(1, len(interest_num)):
        fin_idx = (np.concatenate((fin_idx[0], idx[interest_num[i]][0])),)

    fin_idx = fin_idx[0]
    dataset.targets = labels[fin_idx]
    dataset.data = dataset.data[fin_idx]
    dataset.targets, _ = modify_target(dataset.targets, interest_num)
    return dataset


def ToQuantumData(tensor, img_size):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    data = tensor.to(device)
    input_vec = data.view(-1)
    vec_len = input_vec.size()[0]
    input_matrix = torch.zeros(vec_len, vec_len)
    input_matrix[0] = input_vec
    input_matrix = np.float64(input_matrix.transpose(0, 1))
    u, s, v = np.linalg.svd(input_matrix)
    output_matrix = torch.tensor(np.dot(u, v))
    output_data = output_matrix[:, 0].view(1, img_size, img_size)
    return output_data


def ToQuantumMatrix(tensor):
    device = torch.device("cpu")
    data = tensor.to(device)
    input_vec = data.view(-1)
    vec_len = input_vec.size()[0]
    input_matrix = torch.zeros(vec_len, vec_len)
    input_matrix[0] = input_vec
    input_matrix = np.float64(input_matrix.transpose(0, 1))
    u, s, v = np.linalg.svd(input_matrix)
    output_matrix = torch.tensor(np.dot(u, v))
    return output_matrix


def data_pre_pro(img, img_size, drawing=False, verbose=True):
    """
    T1: Downsample the image from 28*28 to 4*4
    T2: Convert classical data to quantum data which
        can be encoded to the quantum states (amplitude)
    Process data by hand, we can also integrate ToQuantumData into transform
    """
    # Print original figure
    img = img
    npimg = img.numpy()
    if drawing:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    # Print resized figure
    image = np.asarray(npimg[0] * 255, np.uint8)
    im = Image.fromarray(image, mode="L")
    im = im.resize((4, 4), Image.BILINEAR)
    if drawing:
        plt.imshow(im, cmap='gray',)
        plt.show()
    # Converting classical data to quantum data
    trans_to_tensor = transforms.ToTensor()
    if verbose:
        print("Classical Data: {}".format(trans_to_tensor(im).flatten()))
        print("Quantum Data: {}".format(ToQuantumData(
            trans_to_tensor(im), img_size).flatten()))

    return ToQuantumMatrix(trans_to_tensor(im)), ToQuantumData(trans_to_tensor(im), img_size)


def fire_ibmq(circuit, shots, Simulation=False, backend_name='ibmq_qasm_simulator', quiet=False):
    """
    Function: fire_ibmq from Listing 6
    Note: used for execute quantum circuit using
          simulation or ibm quantum processor
    Parameters: (1) quantum circuit;
                (2) number of shots;
                (3) simulation or quantum processor;
                (4) backend name if quantum processor.
            """
    count_set = []
    if not Simulation:
        provider = IBMQ.get_provider(
            hub='ibm-q-research', group='uni-cali-la-1', project='main')
        backend = provider.get_backend(backend_name)
    else:
        backend = Aer.get_backend('qasm_simulator')
    job_ibm_q = execute(circuit, backend, shots=shots)
    job_monitor(job_ibm_q, quiet=quiet)
    result_ibm_q = job_ibm_q.result()
    counts = result_ibm_q.get_counts()
    return counts


def analyze(counts):
    """
    Function: analyze from Listing 6
    Note: used for analyze the count on states to
        formulate the probability for each qubit
    Parameters: (1) counts returned by fire_ibmq;
    """

    mycount = {}
    for i in range(2):
        mycount[i] = 0
    for k, v in counts.items():
        bits = len(k)
        for i in range(bits):
            if k[bits-1-i] == "1":
                if i in mycount.keys():
                    mycount[i] += v
                else:
                    mycount[i] = v
    return mycount, bits


def ccz(circ, q1, q2, t, aux1):
    """
    Function: ccz from Listing 3
    Note: using the basic Toffoli gates and CZ gate
          to implement ccz gate, which will flip the
          sign of state |111>
    Parameters: (1) quantum circuit;
                (2-3) control qubits;
                (4) target qubits;
                (5) auxiliary qubits.
    """

    # Apply Z-gate to a state controlled by 3 qubits
    circ.ccx(q1, q2, aux1)
    circ.cz(aux1, t)
    # cleaning the aux bit
    circ.ccx(q1, q2, aux1)
    return circ


def cccz(circ, q1, q2, q3, t, aux1, aux2):
    """
    Function: cccz from Listing 3
    Note: using the basic Toffoli gates and CZ gate
        to implement cccz gate, which will flip the
        sign of state |1111>
    Parameters: (1) quantum circuit;
        (2-4) control qubits;
        (5) target qubits;
        (6-7) auxiliary qubits."""
    # Apply Z-gate to a state controlled by 4 qubits
    circ.ccx(q1, q2, aux1)
    circ.ccx(q3, aux1, aux2)
    circ.cz(aux2, t)
    # cleaning the aux bits
    circ.ccx(q3, aux1, aux2)
    circ.ccx(q1, q2, aux1)
    return circ


def ccccx(circ, q1, q2, q3, q4, t, aux1, aux2):
    """
    Function: cccz from Listing 4
    Note: using the basic Toffoli gate to implement ccccx
          gate. It is used to switch the quantum states
          of |11110> and |11111>.
    Parameters: (1) quantum circuit;
                (2-5) control qubits;
                (6) target qubits;
                (7-8) auxiliary qubits.
    """
    circ.ccx(q1, q2, aux1)
    circ.ccx(q3, q4, aux2)
    circ.ccx(aux2, aux1, t)
    # cleaning the aux bits
    circ.ccx(q3, q4, aux2)
    circ.ccx(q1, q2, aux1)
    return circ
