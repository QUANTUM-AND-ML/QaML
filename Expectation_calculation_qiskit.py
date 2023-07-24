import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.quantum_info import Kraus, SuperOp
from qiskit_aer import AerSimulator, noise
from qiskit import QuantumCircuit, transpile
from qiskit.tools.visualization import plot_histogram
import random

class Pst_calculation_qiskit:
    def __init__(self, circuit_list , phase_list = [], number_of_qubits = 7):
        self.circuit_list = circuit_list
        self.phase_list = phase_list
        self.number_of_qubits = number_of_qubits

    def pst_calculation_qiskit(self):
        self.circuit = QuantumCircuit(self.number_of_qubits)
        #self.number_rz = 0
        for gate in self.circuit_list:
            if gate[0] == 't':
                self.circuit.t(gate[1])
                #self.number_rz += 1
            elif gate[0] == 'h':
                self.circuit.h(gate[1])
            elif gate[0] == 'i':
                self.circuit.i(gate[1])
            elif gate[0] == 'cx':
                self.circuit.cx(gate[1],gate[2])
            else:
                print('beyond the set of gate operations')
        self.circuit.barrier(i for i in range(self.number_of_qubits))
        for gate in self.circuit_list[::-1]:
            if gate[0] == 't':
                self.circuit.tdg(gate[1])
                #self.number_rz += 1
            elif gate[0] == 'h':
                self.circuit.h(gate[1])
            elif gate[0] == 'i':
                self.circuit.i(gate[1])
            elif gate[0] == 'cx':
                self.circuit.cx(gate[1],gate[2])
            else:
                print('beyond the set of gate operations')
        self.circuit.measure_all()
        return self.circuit

def expectation_calculation_qiskit(circuit_list, number_of_qubits , noise_model):
    circuit = QuantumCircuit(number_of_qubits)
    for gate in circuit_list:
        if gate[0] == 't':
            circuit.t(gate[1])
            # self.number_rz += 1
        elif gate[0] == 'h':
            circuit.h(gate[1])
        elif gate[0] == 'i':
            circuit.i(gate[1])
        elif gate[0] == 'cx':
            circuit.cx(gate[1], gate[2])
        else:
            print('beyond the set of gate operations')
    circuit.measure_all()
    print(circuit)
    # Simulation of the quantum circuit
    sim_noise = AerSimulator(noise_model=noise_model)

    # Transpile circuit for noisy basis gates
    circ_CNNs_noise = transpile(circuit, sim_noise)

    result = sim_noise.run(circ_CNNs_noise, shots=1000).result()

    #counts = result.get_counts(0)

    #print(counts)

    #simulator = Aer.get_backend('statevector_simulator')

    # Perform circuit simulation
    #job = execute(circuit, simulator)

    # Obtain simulation result
    #result = job.result()

    valcount = 0 # We store the expectation values here

    key = np.array(list(result.get_counts().keys()))  # output bit strings
    #print(key)
    val = np.array(list(result.get_counts().values()))  # output bit strings probabilities

    #print(val)
    # Calculation of the rescaled expectation values
    for i in range(len(key)):
        if key[i][0] == '1':
            valcount += (val[i] / 1000)
    return valcount
