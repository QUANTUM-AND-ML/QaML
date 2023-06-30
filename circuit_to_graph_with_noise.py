

#默认所有输入量子态为|0>，最后每个比特都是pauliZ测量测量是|0>或者|1>
class Dataset_graph:
    #node_type不同位置为一分别表示初始输入比特、测量、rz、x、sx、cx(CNOT)
    node_type = [0, 0, 0, 0, 0, 0]
    first_qubit_t1_and_t2 = [0, 0]
    second_qubit_t1_and_t2 = [0, 0]
    gate_error = [0]
    readout01 = [0]
    readout10 = [0]
    gate_index = [0]
    #图的数据
    #Feature_vector = []
    #Adjacency_matrix = [[],[]]
    #Label = []

    def __init__(self,circuit_list, phase_list):
        '''
        :param circuit_list: 输入量子线路的列表形式, Example: [('ry', 0), ('rz', 1), ('cx', 0, 1)];
        :param phase_list: 输入量子线路的相位列表, phase: [0.5, 0.6];
        '''
        self.circuit_list = circuit_list
        self.phase_list = phase_list
    '''
    :param circuit_list: 输入量子线路的列表形式和相位列表，Example: [('ry', 0), ('rz', 1), ('cx', 0, 1)] and phase: [0.5, 0.6];
    :return:输出图神经网络需要的图的信息和初始化的特征向量（包括特折矢量Vec，邻接矩阵A，每个节点特征值Fea）
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

        #中间变量
        self.node_type = [0, 0, 0, 0, 0, 0]
        self.first_qubit_t1_and_t2 = [0, 0]
        self.second_qubit_t1_and_t2 = [0, 0]
        self.gate_error = [0]
        self.readout01 = [0]
        self.readout10 = [0]
        self.gate_index = [0]
        self.cnot_label = [0, 0] # Marking of cnot acting qubit and acted qubit

        #图的参数信息
        self.Feature_vector = []
        self.Adjacency_matrix = [[], []]

        self.gate_qubit = []
        self.qubit_node_gate_index = []
        for i in range(self.number_of_qubits):
            self.gate_qubit.append(0) # Intermediate variables
            self.qubit_node_gate_index.append(0) # Serial number of the node


    def circuit_to_graph(self):
        #初始量子位，默认输入都是|0>
        for i in range(self.number_of_qubits):
            self.node_type[0] = 1
            self.gate_qubit[i] = 1
            self.Feature_vector.append(self.node_type[:] + self.gate_qubit[:] + self.cnot_label[:] + self.first_qubit_t1_and_t2[:]
                                      + self.second_qubit_t1_and_t2[:] + self.gate_error[:] + self.readout10[:] + self.readout01[:] +self.gate_index[:])
            self.qubit_node_gate_index[i] = i
            # 图节点增加1
            self.gate_index[0] += 1
            #还原图数据的中间列表
            self.node_type[0] = 0
            self.gate_qubit[i] = 0

        #量子线路转化为图
        for gate in self.circuit_list:
            if gate[0] == 't':
                self.node_type[2] = 1
                self.gate_qubit[gate[1]] = 1
                #赋值门的t1和t2
                self.first_qubit_t1_and_t2[0] = self.T1[gate[1]]
                self.first_qubit_t1_and_t2[1] = self.T2[gate[1]]
                for lst in self.gateErrors:
                    if lst[0] == 't':
                        self.gate_error[0] = lst[gate[1] + 1]
                #if self.gate_error[0] == 0:
                    #print('t门门误差出现错误')
                #输出图节点的特征矢量
                self.Feature_vector.append(self.node_type[:] + self.gate_qubit[:] + self.cnot_label[:] + self.first_qubit_t1_and_t2[:]
                                        + self.second_qubit_t1_and_t2[:] + self.gate_error[:] + self.readout10[:] + self.readout01[:] + self.gate_index[:])
                #图节点上加入节点，重新增加邻接矩阵的连接情况
                self.Adjacency_matrix[0].append(self.qubit_node_gate_index[gate[1]])
                self.Adjacency_matrix[0].append(self.gate_index[0])
                self.Adjacency_matrix[1].append(self.gate_index[0])
                self.Adjacency_matrix[1].append(self.qubit_node_gate_index[gate[1]])
                self.qubit_node_gate_index[gate[1]] = self.gate_index[0]
                #图节点增加1
                self.gate_index[0] += 1
                # 还原图数据的中间列表
                self.first_qubit_t1_and_t2[0] = 0
                self.first_qubit_t1_and_t2[1] = 0
                self.node_type[2] = 0
                self.gate_qubit[gate[1]] = 0
                self.gate_error[0] = 0
            elif gate[0] == 'h':
                self.node_type[3] = 1
                self.gate_qubit[gate[1]] = 1
                # 赋值门的t1和t2
                self.first_qubit_t1_and_t2[0] = self.T1[gate[1]]
                self.first_qubit_t1_and_t2[1] = self.T2[gate[1]]
                for lst in self.gateErrors:
                    if lst[0] == 'h':
                        self.gate_error[0] = lst[gate[1] + 1]
                #if self.gate_error[0] == 0:
                    #print('h门门误差出现错误')
                # 输出图节点的特征矢量
                self.Feature_vector.append(self.node_type[:] + self.gate_qubit[:] + self.cnot_label[:] + self.first_qubit_t1_and_t2[:]
                                           + self.second_qubit_t1_and_t2[:] + self.gate_error[:] + self.readout10[
                                                                                                   :] + self.readout01[
                                                                                                        :] + self.gate_index[
                                                                                                             :])
                # 图节点上加入节点，重新增加邻接矩阵的连接情况
                self.Adjacency_matrix[0].append(self.qubit_node_gate_index[gate[1]])
                self.Adjacency_matrix[0].append(self.gate_index[0])
                self.Adjacency_matrix[1].append(self.gate_index[0])
                self.Adjacency_matrix[1].append(self.qubit_node_gate_index[gate[1]])
                self.qubit_node_gate_index[gate[1]] = self.gate_index[0]
                # 图节点增加1
                self.gate_index[0] += 1
                # 还原图数据的中间列表
                self.first_qubit_t1_and_t2[0] = 0
                self.first_qubit_t1_and_t2[1] = 0
                self.node_type[3] = 0
                self.gate_qubit[gate[1]] = 0
                self.gate_error[0] = 0
            elif gate[0] == 'i':
                self.node_type[4] = 1
                self.gate_qubit[gate[1]] = 1
                # 赋值门的t1和t2
                self.first_qubit_t1_and_t2[0] = self.T1[gate[1]]
                self.first_qubit_t1_and_t2[1] = self.T2[gate[1]]
                for lst in self.gateErrors:
                    if lst[0] == 'i':
                        self.gate_error[0] = lst[gate[1] + 1]
                #if self.gate_error[0] == 0:
                    #print('i门门误差出现错误')
                # 输出图节点的特征矢量
                self.Feature_vector.append(self.node_type[:] + self.gate_qubit[:] + self.cnot_label[:] + self.first_qubit_t1_and_t2[:]
                                           + self.second_qubit_t1_and_t2[:] + self.gate_error[:] + self.readout10[
                                                                                                   :] + self.readout01[
                                                                                                        :] + self.gate_index[
                                                                                                             :])
                # 图节点上加入节点，重新增加邻接矩阵的连接情况
                self.Adjacency_matrix[0].append(self.qubit_node_gate_index[gate[1]])
                self.Adjacency_matrix[0].append(self.gate_index[0])
                self.Adjacency_matrix[1].append(self.gate_index[0])
                self.Adjacency_matrix[1].append(self.qubit_node_gate_index[gate[1]])
                self.qubit_node_gate_index[gate[1]] = self.gate_index[0]
                # 图节点增加1
                self.gate_index[0] += 1
                # 还原图数据的中间列表
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

                # 赋值门的t1和t2
                self.first_qubit_t1_and_t2[0] = self.T1[gate[1]]
                self.first_qubit_t1_and_t2[1] = self.T2[gate[1]]
                self.second_qubit_t1_and_t2[0] = self.T1[gate[2]]
                self.second_qubit_t1_and_t2[1] = self.T2[gate[2]]
                for lst in self.gateErrors:
                    if lst[0] == 'cx':
                        self.gate_error[0] = (lst[gate[1] + 1] +lst[gate[2] + 1]) / 2
                #if self.gate_error[0] == 0:
                    #print('cx门门误差出现错误')
                # 输出图节点的特征矢量
                self.Feature_vector.append(self.node_type[:] + self.gate_qubit[:] + self.cnot_label[:] + self.first_qubit_t1_and_t2[:] + self.second_qubit_t1_and_t2[:] + self.gate_error[:] + self.readout10[
                                                                                                   :] + self.readout01[:] + self.gate_index[:])
                # 图节点上加入节点，重新增加邻接矩阵的连接情况
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
                # 图节点增加1
                self.gate_index[0] += 1
                # 还原图数据的中间列表
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
                print('存在超出门集合的门')

        #量子线路测量，默认PauliZ测量
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
            # 图节点增加1
            self.gate_index[0] += 1
            # 还原图数据的中间列表
            self.node_type[0] = 0
            self.gate_qubit[i] = 0
            self.readout01[0] = 0
            self.readout10[0] = 0

        return self.Feature_vector, self.Adjacency_matrix, self.Label


'''
circuit = [('h', 0), ('h', 1), ('h', 2), ('h', 1), ('h', 2), ('h', 0), ('h', 1), ('t', 2), ('cx', 1, 0), ('cx', 1, 2),('t', 1), ('cx', 2, 1), ('cx', 2, 1)]
#circuit = [('x', 0), ('rz', 1), ('cx', 0, 1), ('sx',1), ('cx', 0, 6), ('cx', 3, 4), ('cx', 2, 5),('rz', 2), ('rz', 4), ('x', 6)]
print(circuit)
phase = [0.5, 0.5, 0.6, 0.6]
number_of_qubit = 3
T1 = [152.29, 170.6, 132.01, 193.66, 29.75, 157.29, 195.66]
T2 = [104.45, 48.34, 112.84, 289.05, 68.53, 137.14, 262.45]
P01 = [0.031, 0.0222, 0.0282, 0.018, 0.023, 0.0264, 0.0072]
P10 = [0.0226, 0.0236, 0.023, 0.013, 0.0164, 0.0208, 0.0054]
gateErrors = [['t', 0.0001836, 0.0003291, 0.0002153, 0.0002381, 0.0003141, 0.0002563, 0.0003357],
              ['h', 0.0001836, 0.0003291, 0.0002153, 0.0002381, 0.0003141, 0.0002563, 0.0003357],
              ['i', 0.0001836, 0.0003291, 0.0002153, 0.0002381, 0.0003141, 0.0002563, 0.0003357],
              ['cx', 0.00735, 0.00769667, 0.00694, 0.00835, 0.01169, 0.0098133, 0.00984]]
d1 = Dataset_graph_nqubits(circuit, phase, T1, T2, P01, P10, gateErrors, number_of_qubit, 1)
feature, adj , label = d1.circuit_to_graph()
for i in range(2*number_of_qubit + len(circuit)):
    print(feature[i][15 + number_of_qubit], feature[i])

print(adj[0])
print(adj[1])
print(label)
'''




