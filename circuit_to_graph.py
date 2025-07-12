# All input quantum states are initialized as|0>，and the final measurement of each qubit is in the computational basis, resulting in either|0> or |1>
class Dataset_graph:
    # The node_type at different positions represents the following operations：input, measurement, rz, x, sx, cx(CNOT)
    node_type = [0, 0, 0, 0, 0, 0]
    first_qubit_t1_and_t2 = [0, 0]
    second_qubit_t1_and_t2 = [0, 0]
    gate_error = [0]
    readout01 = [0]
    readout10 = [0]
    gate_index = [0]

    def __init__(self,circuit_list, phase_list):
        '''
        :param circuit_list: The list representation of the input quantum circuit, Example: [('ry', 0), ('rz', 1), ('cx', 0, 1)];
        :param phase_list: The list representation of the phase of the input quantum circuit, phase: [0.5, 0.6];
        '''
        self.circuit_list = circuit_list
        self.phase_list = phase_list
    '''
    :param circuit_list: The list representation of the input quantum circuit and the list of phases，Example: [('ry', 0), ('rz', 1), ('cx', 0, 1)] and phase: [0.5, 0.6];
    :return: The information of the graph required for the Graph Neural Network (GNN) output and the initialization of feature vectors (including the feature vector Vec, the adjacency matrix A, and the feature values Fea for each node)
    '''

    def circuit_to_graph(self):
        pass

class Dataset_graph_nqubits:

    def __init__(self, circuit_list, T1, T2, P10, P01, gateErrors, number_of_qubits = 16, label = 0):
        self.circuit_list = circuit_list
        self.T1 = T1
        self.T2 = T2
        self.P01 = P01
        self.P10 = P10
        self.gateErrors = gateErrors
        self.number_of_qubits = number_of_qubits
        self.Label = label

        # Intermediate variables
        self.node_type = [0, 0, 0, 0, 0, 0]
        self.first_qubit_t1_and_t2 = [0, 0]
        self.second_qubit_t1_and_t2 = [0, 0]
        self.gate_error = [0]
        self.readout01 = [0]
        self.readout10 = [0]
        self.gate_index = [0]
        self.cnot_label = [0, 0] # Marking of cnot acting qubit and acted qubit

        # Graph parameters information
        self.Feature_vector = []
        self.Adjacency_matrix = [[], []]

        self.gate_qubit = []
        self.qubit_node_gate_index = []
        for i in range(self.number_of_qubits):
            self.gate_qubit.append(0) # Intermediate variables
            self.qubit_node_gate_index.append(0) # Serial number of the node


    def circuit_to_graph(self):
        # Initial quantum state, default inputs are all|0>
        for i in range(self.number_of_qubits):
            self.node_type[0] = 1
            self.gate_qubit[i] = 1
            self.Feature_vector.append(self.node_type[:] + self.gate_qubit[:] + self.cnot_label[:] + self.first_qubit_t1_and_t2[:]
                                      + self.second_qubit_t1_and_t2[:] + self.gate_error[:] + self.readout10[:] + self.readout01[:] +self.gate_index[:])
            self.qubit_node_gate_index[i] = i
            # Add 1 to the graph nodes
            self.gate_index[0] += 1
            # Restore the intermediate list of graph data
            self.node_type[0] = 0
            self.gate_qubit[i] = 0

        # Transform the quantum circuit into a graph
        for gate in self.circuit_list:
            if gate[0] == 't':
                self.node_type[2] = 1
                self.gate_qubit[gate[1]] = 1
                # T1 and t2 of assignment gates
                self.first_qubit_t1_and_t2[0] = self.T1[gate[1]]
                self.first_qubit_t1_and_t2[1] = self.T2[gate[1]]
                for lst in self.gateErrors:
                    if lst[0] == 't':
                        self.gate_error[0] = lst[gate[1] + 1]

                # Output the feature vectors of graph nodes
                self.Feature_vector.append(self.node_type[:] + self.gate_qubit[:] + self.cnot_label[:] + self.first_qubit_t1_and_t2[:]
                                        + self.second_qubit_t1_and_t2[:] + self.gate_error[:] + self.readout10[:] + self.readout01[:] + self.gate_index[:])
                # Add nodes to the graph nodes to re increase the connection of the Adjacency matrix
                self.Adjacency_matrix[0].append(self.qubit_node_gate_index[gate[1]])
                self.Adjacency_matrix[0].append(self.gate_index[0])
                self.Adjacency_matrix[1].append(self.gate_index[0])
                self.Adjacency_matrix[1].append(self.qubit_node_gate_index[gate[1]])
                self.qubit_node_gate_index[gate[1]] = self.gate_index[0]
                # Figure node added by 1
                self.gate_index[0] += 1
                # Restore the middle list of graph data
                self.first_qubit_t1_and_t2[0] = 0
                self.first_qubit_t1_and_t2[1] = 0
                self.node_type[2] = 0
                self.gate_qubit[gate[1]] = 0
                self.gate_error[0] = 0
            elif gate[0] == 'h':
                self.node_type[3] = 1
                self.gate_qubit[gate[1]] = 1
                # T1 and t2 of assignment gates
                self.first_qubit_t1_and_t2[0] = self.T1[gate[1]]
                self.first_qubit_t1_and_t2[1] = self.T2[gate[1]]
                for lst in self.gateErrors:
                    if lst[0] == 'h':
                        self.gate_error[0] = lst[gate[1] + 1]
                
                # Output the feature vectors of graph nodes
                self.Feature_vector.append(self.node_type[:] + self.gate_qubit[:] + self.cnot_label[:] + self.first_qubit_t1_and_t2[:]
                                           + self.second_qubit_t1_and_t2[:] + self.gate_error[:] + self.readout10[
                                                                                                   :] + self.readout01[
                                                                                                        :] + self.gate_index[
                                                                                                             :])
                # Add nodes to the graph nodes to re increase the connection of the Adjacency matrix
                self.Adjacency_matrix[0].append(self.qubit_node_gate_index[gate[1]])
                self.Adjacency_matrix[0].append(self.gate_index[0])
                self.Adjacency_matrix[1].append(self.gate_index[0])
                self.Adjacency_matrix[1].append(self.qubit_node_gate_index[gate[1]])
                self.qubit_node_gate_index[gate[1]] = self.gate_index[0]
                # Figure node added by 1
                self.gate_index[0] += 1
                # Restore the middle list of graph data
                self.first_qubit_t1_and_t2[0] = 0
                self.first_qubit_t1_and_t2[1] = 0
                self.node_type[3] = 0
                self.gate_qubit[gate[1]] = 0
                self.gate_error[0] = 0
            elif gate[0] == 'i':
                self.node_type[4] = 1
                self.gate_qubit[gate[1]] = 1
                # T1 and t2 of assignment gates
                self.first_qubit_t1_and_t2[0] = self.T1[gate[1]]
                self.first_qubit_t1_and_t2[1] = self.T2[gate[1]]
                for lst in self.gateErrors:
                    if lst[0] == 'i':
                        self.gate_error[0] = lst[gate[1] + 1]
                
                # Output the feature vectors of graph nodes
                self.Feature_vector.append(self.node_type[:] + self.gate_qubit[:] + self.cnot_label[:] + self.first_qubit_t1_and_t2[:]
                                           + self.second_qubit_t1_and_t2[:] + self.gate_error[:] + self.readout10[
                                                                                                   :] + self.readout01[
                                                                                                        :] + self.gate_index[
                                                                                                             :])
                # Add nodes to the graph nodes to re increase the connection of the Adjacency matrix
                self.Adjacency_matrix[0].append(self.qubit_node_gate_index[gate[1]])
                self.Adjacency_matrix[0].append(self.gate_index[0])
                self.Adjacency_matrix[1].append(self.gate_index[0])
                self.Adjacency_matrix[1].append(self.qubit_node_gate_index[gate[1]])
                self.qubit_node_gate_index[gate[1]] = self.gate_index[0]
                # Figure node added by 1
                self.gate_index[0] += 1
                # Restore the middle list of graph data
                self.first_qubit_t1_and_t2[0] = 0
                self.first_qubit_t1_and_t2[1] = 0
                self.node_type[4] = 0
                self.gate_qubit[gate[1]] = 0
                self.gate_error[0] = 0
            elif gate[0] == 'cx':
                self.node_type[5] = 1
                self.gate_qubit[gate[1]] = 1
                self.gate_qubit[gate[2]] = 1
                if gate[1] < gate[2]:
                    self.cnot_label = [1, 0]
                elif gate[1] > gate[2]:
                    self.cnot_label = [0, 1]
                else:
                    print('CNOT control qubit and controlled qubit out of error')

                # T1 and t2 of assignment gates
                self.first_qubit_t1_and_t2[0] = self.T1[gate[1]]
                self.first_qubit_t1_and_t2[1] = self.T2[gate[1]]
                self.second_qubit_t1_and_t2[0] = self.T1[gate[2]]
                self.second_qubit_t1_and_t2[1] = self.T2[gate[2]]
                for lst in self.gateErrors:
                    if lst[0] == 'cx':
                        self.gate_error[0] = (lst[gate[1] + 1] +lst[gate[2] + 1]) / 2
                
                # Output the feature vectors of graph nodes
                self.Feature_vector.append(self.node_type[:] + self.gate_qubit[:] + self.cnot_label[:] + self.first_qubit_t1_and_t2[:] + self.second_qubit_t1_and_t2[:] + self.gate_error[:] + self.readout10[
                                                                                                   :] + self.readout01[:] + self.gate_index[:])
                # Add nodes to the graph nodes to re increase the connection of the Adjacency matrix
                self.Adjacency_matrix[0].append(self.qubit_node_gate_index[gate[1]])
                self.Adjacency_matrix[0].append(self.gate_index[0])
                self.Adjacency_matrix[1].append(self.gate_index[0])
                self.Adjacency_matrix[1].append(self.qubit_node_gate_index[gate[1]])
                self.Adjacency_matrix[0].append(self.qubit_node_gate_index[gate[2]])
                self.Adjacency_matrix[0].append(self.gate_index[0])
                self.Adjacency_matrix[1].append(self.gate_index[0])
                self.Adjacency_matrix[1].append(self.qubit_node_gate_index[gate[2]])
                self.qubit_node_gate_index[gate[1]] = self.gate_index[0]
                self.qubit_node_gate_index[gate[2]] = self.gate_index[0]
                # Figure node added by 1
                self.gate_index[0] += 1
                # Restore the middle list of graph data
                self.cnot_label = [0, 0]
                self.first_qubit_t1_and_t2[0] = 0
                self.first_qubit_t1_and_t2[1] = 0
                self.second_qubit_t1_and_t2[0] = 0
                self.second_qubit_t1_and_t2[1] = 0
                self.node_type[5] = 0
                self.gate_qubit[gate[1]] = 0
                self.gate_qubit[gate[2]] = 0
                self.gate_error[0] = 0
            else:
                print('Gates with a set of super exits')

        # Quantum circuit measurement
        for i in range(self.number_of_qubits):
            self.node_type[1] = 1
            self.gate_qubit[i] = 1
            self.readout01[0] = self.P01[i]
            self.readout10[0] = self.P10[i]
            self.Feature_vector.append(self.node_type[:] + self.gate_qubit[:] + self.cnot_label[:] + self.first_qubit_t1_and_t2[:]
                                       + self.second_qubit_t1_and_t2[:] + self.gate_error[:] + self.readout10[
                                                                                               :] + self.readout01[
                                                                                                    :] + self.gate_index[
                                                                                                         :])
            self.Adjacency_matrix[0].append(self.qubit_node_gate_index[i])
            self.Adjacency_matrix[0].append(self.gate_index[0])
            self.Adjacency_matrix[1].append(self.gate_index[0])
            self.Adjacency_matrix[1].append(self.qubit_node_gate_index[i])
            self.qubit_node_gate_index[i] = self.gate_index[0]
            # Figure node added by 1
            self.gate_index[0] += 1
            # Restore the middle list of graph data
            self.node_type[0] = 0
            self.gate_qubit[i] = 0
            self.readout01[0] = 0
            self.readout10[0] = 0

        return self.Feature_vector, self.Adjacency_matrix, self.Label
