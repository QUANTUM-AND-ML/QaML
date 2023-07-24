from qiskit import QuantumCircuit, assemble, Aer
import random
import numpy as np
import matplotlib.pyplot as plt

def CNNs_generate_datas(qubits, Number_of_gates_of_per_qubit):
    flag = 0
    N = qubits # Number of qubits
    P = Number_of_gates_of_per_qubit # Number of gates per qubit
    circuit_list = []
    circ = np.zeros((N,P))
    #qc = QuantumCircuit(N)

    for j in range(P):  # Loop over the gates layers
        col = np.zeros(N)
        countarget = 0
        for i in range(N):
            if col[i] != 3:  # It is not the target of the CX-gate
                listemp = []
                if flag == 0:  # There isn't a CX-gate in this layer of gates
                    r = random.randint(0, 2)  # Random choice between H, T and the control of the CX-gate
                    listemp.append(r)
                    if r == 2:  # If it is the control of the CX-gate
                        r2 = random.randint(1, N - 1)
                        t = (i + r2) % N  # This is the target of the CX-gate
                        listemp.append(t)
                else:  # If the CX-gate has been defined, only 1-qubit gate can be added in this layer of gates
                    r = random.randint(0, 1)
                    listemp.append(r)
                col[i] = listemp[0]
                if len(listemp) == 2:
                    flag = 1  # Now there is a CX-gate in this layer of gates
                    col[listemp[1]] = 3
        for i in range(N):  # We build the j-th layer of gates
            if col[i] == 0:
                #qc.t(i)
                circuit_list.append(('t', i))
            elif col[i] == 1:
                #qc.h(i)
                circuit_list.append(('h', i))
            elif col[i] == 2:
                #qc.cx(i, t)
                circuit_list.append(('cx', i, t))

        # We store the gates that we have generated in a list
        circ[:, j] = col
        flag = 0
    #print(circ)
    #qc.draw(output='mpl')
    #plt.show()
    return circuit_list, circ
