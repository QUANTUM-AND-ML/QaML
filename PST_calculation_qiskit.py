import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.quantum_info import Kraus, SuperOp
from qiskit_aer import AerSimulator, noise
from qiskit import QuantumCircuit, transpile
from qiskit.tools.visualization import plot_histogram
import random

from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)

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
                print('超出门集门出现')
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
                print('超出门集门出现')
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
            print('超出门集门出现')
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

    # 执行线路仿真
    #job = execute(circuit, simulator)

    # 获取仿真结果
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
'''
number_of_qubit = 5
circuit_list = [('h', 0), ('h', 3), ('h', 4)]
valcount = expectation_calculation_qiskit(circuit_list, number_of_qubit)
print(valcount)
'''






'''
# Measurement miss-assignement probabilities
p0given1 = 0.1
p1given0 = 0.05

ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]])

# Create an empty noise model
noise_model = NoiseModel()

# Add depolarizing error to all single qubit u1, u2, u3 gates
error1 = depolarizing_error(0.05, 1)
error2 = depolarizing_error(0.1, 2)
noise_model.add_all_qubit_quantum_error(error1, ['h', 'x', 'u3'])
noise_model.add_all_qubit_quantum_error(error2, ['cx'])
# Print noise model info
print(noise_model)
'''

#circuit = [('x', 0), ('rz', 1),('x', 1), ('cx', 0, 1), ('sx',1), ('cx', 0, 6), ('cx', 3, 4), ('cx', 2, 5),('rz', 2), ('rz', 4), ('x', 6)]
#print(circuit)
#phase = [0.1, 0.2, 0.3]
#print(phase)
#cir1 = Pst_calculation_qiskit(circuit, phase, 7)
#cir1_circuit = cir1.pst_calculation_qiskit()
#cir1_circuit.draw(output='mpl')
#plt.show()

# Create noisy simulator backend
#sim_noise = AerSimulator(noise_model = noise_model)

# Transpile circuit for noisy basis gates
#circ_tnoise = transpile(cir1_circuit, sim_noise)

#result_ideal = sim_noise.run(cir1_circuit).result()
#plot_histogram(result_ideal.get_counts(0))
#plt.show()

'''
# 量子线路的噪声参数
number_of_qubit = 7
T1 = [197.79, 158.07, 278.36, 223.29, 109.91, 236.63, 193.96]
T2 = [97.02, 48.06, 83.1, 211, 126.96, 188.44, 291.83]
P01 = [0.031, 0.0222, 0.0282, 0.018, 0.023, 0.0264, 0.0072]
P10 = [0.0226, 0.0236, 0.023, 0.013, 0.0164, 0.0208, 0.0054]
gateErrors = [['t', 0.0001836, 0.0003291, 0.0002153, 0.0002381, 0.0003141, 0.0002563, 0.0003357],
              ['h', 0.0001836, 0.0003291, 0.0002153, 0.0002381, 0.0003141, 0.0002563, 0.0003357],
              ['i', 0.0001836, 0.0003291, 0.0002153, 0.0002381, 0.0003141, 0.0002563, 0.0003357],
              ['cx', 0.00735, 0.00769667, 0.00694, 0.00835, 0.01169, 0.0098133, 0.00984]]
# 退极化噪声添加
# Error probabilities

# Add errors to noise model
noise_model = noise.NoiseModel()
for i in range(number_of_qubit):
    noise_model.add_quantum_error(depolarizing_error(gateErrors[0][i + 1], 1), ['t', 'h', 'i', 'tdg'], [i])
    for k in range(number_of_qubit):
        noise_model.add_quantum_error(depolarizing_error((gateErrors[3][i + 1] + gateErrors[3][k + 1]) / 2, 2),
                                      ['cx'], [i, k])

# 加入T1和T2
# Instruction times (in nanoseconds)
time_reset = 1000  # 1 microsecond
time_measure = 1000  # 1 microsecond
time_t = 100  # virtual gate
time_tdg = 100  # virtual gate
time_h = 50  # (single X90 pulse)
time_i = 50  # (two X90 pulses)
time_cx = 300

# QuantumError objects
errors_reset = [thermal_relaxation_error(t1 * 1000, t2 * 1000, time_reset)
                for t1, t2 in zip(T1, T2)]
errors_measure = [thermal_relaxation_error(t1 * 1000, t2 * 1000, time_measure)
                  for t1, t2 in zip(T1, T2)]
errors_t = [thermal_relaxation_error(t1 * 1000, t2 * 1000, time_t)
             for t1, t2 in zip(T1, T2)]
errors_tdg = [thermal_relaxation_error(t1 * 1000, t2 * 1000, time_tdg)
             for t1, t2 in zip(T1, T2)]
errors_h = [thermal_relaxation_error(t1 * 1000, t2 * 1000, time_h)
            for t1, t2 in zip(T1, T2)]
errors_i = [thermal_relaxation_error(t1 * 1000, t2 * 1000, time_i)
             for t1, t2 in zip(T1, T2)]
errors_cx = [[thermal_relaxation_error(t1a * 1000, t2a * 1000, time_cx).expand(
    thermal_relaxation_error(t1b * 1000, t2b * 1000, time_cx))
    for t1a, t2a in zip(T1, T2)]
    for t1b, t2b in zip(T1, T2)]

# Add errors to noise model
for j in range(number_of_qubit):
    noise_model.add_quantum_error(errors_reset[j], "reset", [j])
    noise_model.add_quantum_error(errors_measure[j], "measure", [j])
    noise_model.add_quantum_error(errors_t[j], "t", [j])
    noise_model.add_quantum_error(errors_tdg[j], "tdg", [j])
    noise_model.add_quantum_error(errors_h[j], "h", [j])
    noise_model.add_quantum_error(errors_i[j], "i", [j])
    for k in range(number_of_qubit):
        noise_model.add_quantum_error(errors_cx[j][k], "cx", [j, k])

# 添加读出错误
# Measurement miss-assignement probabilities
for i in range(number_of_qubit):
    noise_model.add_readout_error(ReadoutError([[1 - P10[i], P10[i]], [P01[i], 1 - P01[i]]]), [i])

print(noise_model)

#circuit_list = [('rz', 0)]

circuit_list = [('h', 0), ('h', 1), ('h', 2), ('h', 3), ('h', 4), ('h', 5), ('h', 6)]
phase_list = []
a = expectation_calculation_qiskit(circuit_list, 7, noise_model)

cir1 = Pst_calculation_qiskit(circuit_list, [], number_of_qubit)
cir1_circuit = cir1.pst_calculation_qiskit()
cir1_circuit.draw(output='mpl')
plt.show()


print((cir1_circuit.depth() + 1) // 2)
#width[0] = cir1_circuit.width()
# 仿真得到实际的PST
# Create noisy simulator backend
sim_noise = AerSimulator(noise_model=noise_model)
# Transpile circuit for noisy basis gates
circ_tnoise = transpile(cir1_circuit, sim_noise)
result_noise = sim_noise.run(circ_tnoise, shots=10000).result()
if result_noise.get_counts(0).get('0000000') == None:
    PST = 0
else:
    PST = result_noise.get_counts(0).get('0000000') / 10000

print('PST:', PST)
'''