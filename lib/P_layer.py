from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.extensions import UnitaryGate
from qiskit.pulse.channels import MeasureChannel

from lib.utils import *


def neg_weight_gate(circ, qubits, aux, state):
    """
    Flips the sign of the given gate. 
    
    1. swap the amplitude of the given state to the state of |1⟩⊗4.
    2. the cccz gate is applied to complete the second step.
    3. amplitude is swap back to the given state.

    Parameters: (1) quantum circuit;
                (2) all qubits, say q0-q3;
                (3) the auxiliary qubits used for cccz
                (4) states, say 1101
    """
    
    idx = 0
    
    # The index of qubits are reversed in terms of states.
    # As shown in the above example: we put X at q2 not the third position.
    state = state[::-1]
    for idx in range(len(state)):
        if state[idx] == '0':
            circ.x(qubits[idx])
    cccz(circ, qubits[0], qubits[1], qubits[2], qubits[3], aux[0], aux[1])
    for idx in range(len(state)):
        if state[idx] == '0':
            circ.x(qubits[idx])


def p_circ_gen(quantum_matrix, weights, flags, params, batch_norm=True):

    # Quantum circuit implementation of the output layer
    # fundamentals, please see our Nature Communication
    # paper (P-LYR) https://arxiv.org/pdf/2006.14815.pdf

    """
    weights = [weight_1_1, weight_1_2, weight_2_1, weight_2_2]
    flags = [norm_flag_1, norm_flag_2]
    para = [norm_para_1, norm_para_2]
    """

    weight_1_1 = weights[0]
    weight_1_2 = weights[1]
    weight_2_1 = weights[2]
    weight_2_2 = weights[3]

    norm_flag_1 = flags[0]
    norm_flag_2 = flags[1]

    norm_para_1 = params[0]
    norm_para_2 = params[1]

    ### Hidden layer ###

    # From Listing 2: create the qubits to hold data
    inp_1 = QuantumRegister(4, "in1_qbit")
    inp_2 = QuantumRegister(4, "in2_qbit")
    circ = QuantumCircuit(inp_1, inp_2)
    data_matrix = quantum_matrix
    circ.append(UnitaryGate(data_matrix, label="Input"), inp_1[0:4])
    circ.append(UnitaryGate(data_matrix, label="Input"), inp_2[0:4])

    # From Listing 3: create auxiliary qubits
    aux = QuantumRegister(2, "aux_qbit")
    circ.add_register(aux)

    # From Listing 4: create output qubits for the first layer (hidden neurons)
    hidden_neurons = QuantumRegister(2, "hidden_qbits")
    circ.add_register(hidden_neurons)

    # From Listing 3: to multiply inputs and weights on quantum circuit
    if weight_1_1.sum() < 0:
        weight_1_1 = weight_1_1*-1
    idx = 0
    for idx in range(weight_1_1.flatten().size()[0]):
        if weight_1_1[idx] == -1:
            # 4 bit binary representation of idx
            state = "{0:b}".format(idx).zfill(4)
            neg_weight_gate(circ, inp_1, aux, state)
            circ.barrier()

    if weight_1_2.sum() < 0:
        weight_1_2 = weight_1_2*-1
    idx = 0
    for idx in range(weight_1_2.flatten().size()[0]):
        if weight_1_2[idx] == -1:
            state = "{0:b}".format(idx).zfill(4)
            neg_weight_gate(circ, inp_2, aux, state)
            circ.barrier()

    # From Listing 4: applying the quadratic function on the weighted sum
    circ.h(inp_1)
    circ.x(inp_1)
    ccccx(circ, inp_1[0], inp_1[1], inp_1[2],
          inp_1[3], hidden_neurons[0], aux[0], aux[1])

    circ.h(inp_2)
    circ.x(inp_2)
    ccccx(circ, inp_2[0], inp_2[1], inp_2[2],
          inp_2[3], hidden_neurons[1], aux[0], aux[1])

    ### output layer ###

    if batch_norm:
        inter_q_1 = QuantumRegister(1, "inter_q_1_qbits")
        norm_q_1 = QuantumRegister(1, "norm_q_1_qbits")
        out_q_1 = QuantumRegister(1, "out_q_1_qbits")
        circ.add_register(inter_q_1, norm_q_1, out_q_1)

        circ.barrier()

        if weight_2_1.sum() < 0:
            weight_2_1 = weight_2_1*-1
        idx = 0
        for idx in range(weight_2_1.flatten().size()[0]):
            if weight_2_1[idx] == -1:
                circ.x(hidden_neurons[idx])
        circ.h(inter_q_1)
        circ.cz(hidden_neurons[0], inter_q_1)
        circ.x(inter_q_1)
        circ.cz(hidden_neurons[1], inter_q_1)
        circ.x(inter_q_1)
        circ.h(inter_q_1)
        circ.x(inter_q_1)

        circ.barrier()

        norm_init_rad = float(norm_para_1.sqrt().arcsin()*2)
        circ.ry(norm_init_rad, norm_q_1)
        if norm_flag_1:
            circ.cx(inter_q_1, out_q_1)
            circ.x(inter_q_1)
            circ.ccx(inter_q_1, norm_q_1, out_q_1)
        else:
            circ.ccx(inter_q_1, norm_q_1, out_q_1)

        for idx in range(weight_2_1.flatten().size()[0]):
            if weight_2_1[idx] == -1:
                circ.x(hidden_neurons[idx])

        circ.barrier()

        inter_q_2 = QuantumRegister(1, "inter_q_2_qbits")
        norm_q_2 = QuantumRegister(1, "norm_q_2_qbits")
        out_q_2 = QuantumRegister(1, "out_q_2_qbits")
        circ.add_register(inter_q_2, norm_q_2, out_q_2)

        circ.barrier()

        if weight_2_2.sum() < 0:
            weight_2_2 = weight_2_2*-1
        idx = 0
        for idx in range(weight_2_2.flatten().size()[0]):
            if weight_2_2[idx] == -1:
                circ.x(hidden_neurons[idx])
        circ.h(inter_q_2)
        circ.cz(hidden_neurons[0], inter_q_2)
        circ.x(inter_q_2)
        circ.cz(hidden_neurons[1], inter_q_2)
        circ.x(inter_q_2)
        circ.h(inter_q_2)
        circ.x(inter_q_2)

        circ.barrier()

        norm_init_rad = float(norm_para_2.sqrt().arcsin()*2)
        circ.ry(norm_init_rad, norm_q_2)
        if norm_flag_2:
            circ.cx(inter_q_2, out_q_2)
            circ.x(inter_q_2)
            circ.ccx(inter_q_2, norm_q_2, out_q_2)
        else:
            circ.ccx(inter_q_2, norm_q_2, out_q_2)

        for idx in range(weight_2_2.flatten().size()[0]):
            if weight_2_2[idx] == -1:
                circ.x(hidden_neurons[idx])

        circ.barrier()

        c_reg = ClassicalRegister(2, "reg")
        circ.add_register(c_reg)
        circ.measure(out_q_1, c_reg[0])
        circ.measure(out_q_2, c_reg[1])

        # print("Output layer created!")

    else:

        c_reg = ClassicalRegister(2, "reg")
        circ.add_register(c_reg)
        circ.barrier() 
        circ.measure(hidden_neurons[0], c_reg[0])
        circ.measure(hidden_neurons[1], c_reg[1])

    return circ


def p_single_circ_gen(quantum_matrix, weights, flags, params, batch_norm=True):

    """
    x = 1 -> class 0; x = 2 -> class 1
    weights = [weight_1_x, weight_2_x,]
    flags = norm_flag_x
    para = norm_para_x
    """

    weight_1_1 = weights[0]
    weight_2_1 = weights[1]

    norm_flag_1 = flags
    norm_para_1 = params

    ### Hidden layer ###

    # From Listing 2: create the qubits to hold data
    inp_1 = QuantumRegister(4, "in1_qbit")
    # inp_2 = QuantumRegister(4, "in2_qbit")

    # circ = QuantumCircuit(inp_1, inp_2)
    
    circ = QuantumCircuit(inp_1)
    data_matrix = quantum_matrix
    circ.append(UnitaryGate(data_matrix, label="Input"), inp_1[0:4])
    # circ.append(UnitaryGate(data_matrix, label="Input"), inp_2[0:4])

    # From Listing 3: create auxiliary qubits
    aux = QuantumRegister(2, "aux_qbit")
    circ.add_register(aux)

    # From Listing 4: create output qubits for the first layer (hidden neurons)
    hidden_neurons = QuantumRegister(2, "hidden_qbits")
    circ.add_register(hidden_neurons)

    # From Listing 3: to multiply inputs and weights on quantum circuit
    if weight_1_1.sum() < 0:
        weight_1_1 = weight_1_1*-1
    idx = 0
    for idx in range(weight_1_1.flatten().size()[0]):
        if weight_1_1[idx] == -1:
            # 4 bit binary representation of idx
            state = "{0:b}".format(idx).zfill(4)
            neg_weight_gate(circ, inp_1, aux, state)
            circ.barrier()

    # From Listing 4: applying the quadratic function on the weighted sum
    circ.h(inp_1)
    circ.x(inp_1)
    ccccx(circ, inp_1[0], inp_1[1], inp_1[2],
          inp_1[3], hidden_neurons[0], aux[0], aux[1])
    
    ### output layer ###
    if batch_norm:

        inter_q_1 = QuantumRegister(1, "inter_q_1_qbits")
        norm_q_1 = QuantumRegister(1, "norm_q_1_qbits")
        out_q_1 = QuantumRegister(1, "out_q_1_qbits")
        circ.add_register(inter_q_1, norm_q_1, out_q_1)

        circ.barrier()

        if weight_2_1.sum() < 0:
            weight_2_1 = weight_2_1*-1
        idx = 0
        for idx in range(weight_2_1.flatten().size()[0]):
            if weight_2_1[idx] == -1:
                circ.x(hidden_neurons[idx])
        circ.h(inter_q_1)
        circ.cz(hidden_neurons[0], inter_q_1)
        circ.x(inter_q_1)
        circ.cz(hidden_neurons[1], inter_q_1)
        circ.x(inter_q_1)
        circ.h(inter_q_1)
        circ.x(inter_q_1)

        circ.barrier()

        norm_init_rad = float(norm_para_1.sqrt().arcsin()*2)
        circ.ry(norm_init_rad, norm_q_1)
        if norm_flag_1:
            circ.cx(inter_q_1, out_q_1)
            circ.x(inter_q_1)
            circ.ccx(inter_q_1, norm_q_1, out_q_1)
        else:
            circ.ccx(inter_q_1, norm_q_1, out_q_1)

        for idx in range(weight_2_1.flatten().size()[0]):
            if weight_2_1[idx] == -1:
                circ.x(hidden_neurons[idx])

    circ.barrier()

    c_reg = ClassicalRegister(1, "reg")
    circ.add_register(c_reg)
    circ.measure(out_q_1, c_reg[0])

    return circ