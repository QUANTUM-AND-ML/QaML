from torch_geometric.data import InMemoryDataset
import random
from circuit_to_graph_with_noise import Dataset_graph_nqubits
from expectation_and_ground_state_energy_calculation import expectation_calculation_qiskit, expectation_calculation_qiskit_with_noise, ground_state_energy_calculation
from Simplification import simplification
from torch_geometric.data import Data
import numpy as np
import torch
import matplotlib.pyplot as plt
from qiskit_aer import noise
import networkx as nx
from qiskit import QuantumCircuit, transpile
from Simplification import *
from CNNS_generate_datas import CNNs_generate_datas
from qiskit.quantum_info import Kraus, SuperOp
from qiskit_aer import AerSimulator
from torch_geometric.data import DataLoader, Batch
from qiskit.tools.visualization import plot_histogram

# Import from Qiskit Aer noise module
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)

# generate 1000 data
class MyDataset(InMemoryDataset):
    def __init__(self, root, transform = None, pre_transform = None):
        super(MyDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'data.pt'


    def process(self):
        data_list = []
        for count in range(10000): # the number of circuits generated

            # generate dataset
            # Noise parameters of Quantum circuit
            number_of_qubit = 7
            if count > 10000 - 2000:
                number_of_qubit = 11

            # Noisy Simulator. The dataset is saved in a folder named data.
            T1 = [197.79, 158.07, 278.36, 223.29, 109.91, 236.63, 193.96,197.79, 158.07, 278.36, 223.29, 109.91, 236.63, 193.96,197.79, 158.07]
            T2 = [97.02, 48.06, 83.1, 211, 126.96, 188.44, 291.83,97.02, 48.06, 83.1, 211, 126.96, 188.44, 291.83,97.02, 48.06]
            # Read error (not real Quantum computer)
            P01 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
            P10 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
            # Self defined gate error (non real Quantum computer)
            gateErrors = [['rz', 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
                          ['x',  0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
                          ['sx', 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
                          ['cx', 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]]

            '''
            # IBM Perth. The dataset is saved in a folder named data_ibm_perth.
            T1 = [197.79, 158.07, 278.36, 223.29, 109.91, 236.63, 193.96]
            T2 = [97.02, 48.06, 83.1, 211, 126.96, 188.44, 291.83]
            P01 = [0.031, 0.0222, 0.0282, 0.018, 0.023, 0.0264, 0.0072]
            P10 = [0.0226, 0.0236, 0.023, 0.013, 0.0164, 0.0208, 0.0054]
            gateErrors = [['rz', 0.0001836, 0.0003291, 0.0002153, 0.0002381, 0.0003141, 0.0002563, 0.0003357],
                          ['x', 0.0001836, 0.0003291, 0.0002153, 0.0002381, 0.0003141, 0.0002563, 0.0003357],
                          ['sx', 0.0001836, 0.0003291, 0.0002153, 0.0002381, 0.0003141, 0.0002563, 0.0003357],
                          ['cx', 0.00735, 0.00769667, 0.00694, 0.00835, 0.01169, 0.0098133, 0.00984]]
            '''
            '''
            # IBM Lagos. The dataset is saved in a folder named data_ibm_lagos.
            T1 = [96, 131.71, 174.23, 139.45, 113.32, 131.19, 93.17]
            T2 = [40.53, 75.18, 160.96, 84.68, 29.5, 64.52, 81.31]
            P01 = [0.018, 0.0192, 0.0094, 0.0166, 0.021, 0.0148, 0.013]
            P10 = [0.0092, 0.0178, 0.0074, 0.0144, 0.0198, 0.019, 0.0152]
            gateErrors = [['rz', 1.813e-4, 1.439e-4, 2.279e-4, 1.988e-4, 2.090e-4, 1.975e-4, 1.845e-4],
                          ['x', 1.813e-4, 1.439e-4, 2.279e-4, 1.988e-4, 2.090e-4, 1.975e-4, 1.845e-4],
                          ['sx', 1.813e-4, 1.439e-4, 2.279e-4, 1.988e-4, 2.090e-4, 1.975e-4, 1.845e-4],
                          ['cx', 0.006553, 0.00769667, 0.00685, 0.006505, 0.00723, 0.0068467, 0.00588]]
            '''
            '''
            # IBM Nairobi. The dataset is saved in a folder named data_ibm_nairobi.
            T1 = [104.92, 141.98, 57.67, 145.77, 99.1, 126.61, 148.77]
            T2 = [26.07, 97.62, 69.22, 59.13, 61.06, 16.25, 105.21]
            P01 = [0.0318, 0.072, 0.0406, 0.0366, 0.0326, 0.0432, 0.041]
            P10 = [0.011, 0.0198, 0.0116, 0.0096, 0.0098, 0.0204, 0.0142]
            gateErrors = [['rz', 2.782e-4, 5.325e-4, 0.00365, 3.363e-4, 2.725e-4, 2.855e-4, 2.028e-4],
                          ['x', 2.782e-4, 5.325e-4, 0.00365, 3.363e-4, 2.725e-4, 2.855e-4, 2.028e-4],
                          ['sx', 2.782e-4, 5.325e-4, 0.00365, 3.363e-4, 2.725e-4, 2.855e-4, 2.028e-4],
                          ['cx', 0.01034, 0.01765, 0.03446, 0.01259, 0.00954, 0.0111767, 0.00696]]
            '''
            '''
            # IBM Jakarta. The dataset is saved in a folder named data_ibm_jakarta.
            T1 = [145.55, 143.21, 110.89, 78.53, 145.55, 128.7, 139.55]
            T2 = [47.91, 28.18, 22.67, 36.52, 47.91, 65.54, 21.58]
            P01 = [0.0282, 0.0396, 0.0252, 0.0254, 0.0282, 0.0646, 0.04]
            P10 = [0.007, 0.0096, 0.0082, 0.0112, 0.007, 0.048, 0.0244]
            gateErrors = [['t', 3.194e-4, 2.136e-4, 2.076e-4, 2.149e-4, 3.194e-4, 2.627e-4, 2.507e-4],
                          ['h', 3.194e-4, 2.136e-4, 2.076e-4, 2.149e-4, 3.194e-4, 2.627e-4, 2.507e-4],
                          ['i', 3.194e-4, 2.136e-4, 2.076e-4, 2.149e-4, 3.194e-4, 2.627e-4, 2.507e-4],
                          ['cx', 0.00806, 0.00872, 0.00914, 0.007755, 0.00806, 0.00616, 0.00577]]
            '''

            # Depolarization noise addition

            # Add errors to noise model
            noise_model = noise.NoiseModel()
            for i in range(number_of_qubit):
                noise_model.add_quantum_error(depolarizing_error(gateErrors[0][i + 1], 1), ['t', 'h', 'i', 'tdg'], [i])
                for k in range(number_of_qubit):
                    noise_model.add_quantum_error(
                        depolarizing_error((gateErrors[3][i + 1] + gateErrors[3][k + 1]) / 2, 2),
                        ['cx'], [i, k])

            # Add T1 and T2
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

            # Add Read Error
            # Measurement miss-assignement probabilities
            for i in range(number_of_qubit):
                noise_model.add_readout_error(ReadoutError([[1 - P10[i], P10[i]], [P01[i], 1 - P01[i]]]), [i])
            
            number_CNNs_t = [0]
            number_CNNs_h = [0]
            number_CNNs_i = [0]
            number_CNNs_cx = [0]
            number_CNNs_gates = [0]
            depth_CNNs = [0]
            width_CNNs = [0]

            number_GAT_t = [0]
            number_GAT_h = [0]
            number_GAT_i = [0]
            number_GAT_cx = [0]
            number_GAT_gates = [0]
            depth_GAT = [0]
            width_GAT = [0]
            wrong = 0
            #number_of_qubit = random.randint(3, 11)
            P = random.randint(5, 11)
            #P = 5
            circuit_CNNs, cic_CNNs_array  = CNNs_generate_datas(number_of_qubit, P)
            circuit_GAT = simplification(circuit_CNNs)
            number_GAT_gates[0] = len(circuit_GAT)
            node_mark = [0]
            #node_mark[0] = len(circuit_GAT) + number_of_qubit * 2 - 1
            node_mark[0] = len(circuit_GAT) + number_of_qubit  + 16 - 1
            count_gate_last_qubit = 0
            count_gate_index_last_qubit = []
            for j in range(len(circuit_GAT)):
                if circuit_GAT[j][0] == 't' and circuit_GAT[j][1] == number_of_qubit - 1:
                    number_GAT_t[0] += 1
                    count_gate_last_qubit += 1
                    count_gate_index_last_qubit.append(j)
                elif circuit_GAT[j][0] == 'h' and circuit_GAT[j][1] == number_of_qubit - 1:
                    number_GAT_h[0] += 1
                    count_gate_last_qubit += 1
                    count_gate_index_last_qubit.append(j)
                elif circuit_GAT[j][0] == 'i' and circuit_GAT[j][1] == number_of_qubit - 1:
                    number_GAT_i[0] += 1
                    count_gate_last_qubit += 1
                    count_gate_index_last_qubit.append(j)
                elif circuit_GAT[j][0] == 'cx' and circuit_GAT[j][2] == number_of_qubit - 1:
                    number_GAT_cx[0] += 1
                    count_gate_last_qubit += 1
                    count_gate_index_last_qubit.append(j)
                else:
                    pass

            if count_gate_last_qubit > 0:
                #node_mark[0] = count_gate_index_last_qubit[round((count_gate_last_qubit - 1) / 2)] + number_of_qubit
                node_mark[0] = count_gate_index_last_qubit[round((count_gate_last_qubit - 1) / 2)] + 16

            # Calculation of expectation value
            expected_value_GAT = expectation_calculation_qiskit(circuit_GAT, number_of_qubit , noise_model)
            expected_value_CNNs = float(expectation_calculation_qiskit(circuit_CNNs, number_of_qubit, noise_model))
            #if round(expected_value_GAT * 1000) != round(expected_value_CNNs * 1000):
                #wrong = wrong + 1
            print('Number：', count)
            print('number of GAT gates:', number_GAT_gates, 'expected value GAT:', expected_value_GAT)
            print('node_index:', node_mark[0])
            if count_gate_last_qubit > 0:
                #print(circuit_GAT[node_mark[0] - number_of_qubit])
                print(circuit_GAT[node_mark[0] - 16])


            d = Dataset_graph_nqubits(circuit_GAT, T1, T2, P10, P01, gateErrors, 16, 0)
            feature, adj, label = d.circuit_to_graph()
            #print(feature)
            #print(label)
            graph_features_lst = number_GAT_cx + number_GAT_h + number_GAT_t
            graph_features = torch.tensor(graph_features_lst, dtype = torch.float)
            print(graph_features)
            x = torch.tensor(feature, dtype = torch.float)
            edge_index = torch.tensor(adj, dtype = torch.long)
            # y = torch.tensor([label for _ in range(14 + number_gates +7)], dtype = torch.float)

            graph_label = round(expected_value_GAT, 3)

            data = Data(x = x, edge_index = edge_index)
            data.y = torch.tensor(graph_label, dtype = torch.float) # Label of the entire image
            data.graph_attr = graph_features
            data.node_index = torch.tensor(node_mark[0], dtype = torch.float)
            #print(data.graph_attr)
            data_list.append(data)
            #print(data)
        print('Number of times of wrong', wrong)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
