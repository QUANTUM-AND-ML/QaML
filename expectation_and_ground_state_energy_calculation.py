import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.quantum_info import Kraus, SuperOp
from qiskit_aer import AerSimulator, noise
from qiskit import QuantumCircuit, transpile
from qiskit.tools.visualization import plot_histogram
import random
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import Gate, Instruction, Parameter
from qiskit.primitives import Estimator
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.quantum_info import SparsePauliOp

# The Hamiltonian of a hydrogen molecule
H2_op = SparsePauliOp.from_list(
    [
        ("IIII", -0.042),
        ("ZIII", 0.178),
        ("IZII", 0.178),
        ("IIZI", -0.243),
        ("IIIZ", -0.243),
        ("ZZII", 0.171),
        ("IIZZ", 0.176),
        ("ZIZI", 0.123),
        ("IZIZ", 0.123),
        ("ZIIZ", 0.168),
        ("IZZI", 0.168),
        ("YXXY", 0.045),
        ("XYYX", 0.045),
        ("YYXX", -0.045),
        ("XXYY", -0.045),
    ]
)

# Calculate the expectation value of the quantum circuit under noiseless conditions.
def expectation_calculation_qiskit(circuit_list, number_of_qubits):
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
    #print(circuit)
    # Simulation of the quantum circuit
    simulator = Aer.get_backend('statevector_simulator')

    valcount = np.zeros((number_of_qubits))  # We store the expectation values here

    key = np.array(list(result.get_counts().keys()))  # output bit strings
    val = np.array(list(result.get_counts().values()))  # output bit strings probabilities

    # Calculation of the rescaled expectation values
    for i in range(len(key)):
        for j in range(len(key[i])):
            if key[i][j] == '1':
                valcount[j] += val[i]
    return valcount[0]

# Calculate the expectation value of the quantum circuit under noisy conditions
def expectation_calculation_qiskit_with_noise(circuit_list, number_of_qubits , noise_model):
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

# Calculate the ground-state energy of the hydrogen molecule for different quantum circuit structures
def ground_state_energy_calculation(circuit_list):
    num_qubits = 4
    circuits = circuit_list
    number_of_parameter =  0
    for gate in circuits:
        if gate[0] != 'cx':
            number_of_parameter += 1

    estimator = Estimator()
    ansatz = QuantumCircuit(num_qubits)

    # Define the parameters
    params = [Parameter(f'theta_{i}') for i in range(number_of_parameter)]

    parameter_label = 0
    for gates in circuits:
        if gates[0] == 'rx':
            ansatz.rx(params[parameter_label], gates[1])
            parameter_label += 1
        elif gates[0] == 'ry':
            ansatz.ry(params[parameter_label], gates[1])
            parameter_label += 1
        elif gates[0] == 'cx':
            ansatz.cx(gates[1], gates[2])
        else:
            print('beyond the set of gate operations')
    print(ansatz)
    #ansatz.draw(output='mpl')
    #plt.show()

    optimizer = SLSQP(maxiter=1000)
    #optimizer = SPSA(maxiter=1000)

    #ansatz.decompose().draw("mpl", style="iqx")
    #plt.show()

    vqe = VQE(estimator, ansatz, optimizer)

    result = vqe.compute_minimum_eigenvalue(H2_op)
    #print(result)
    return result.eigenvalue.real
