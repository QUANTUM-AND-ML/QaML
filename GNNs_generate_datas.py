import random


def random_selection(numbers):
    set = [0, 1, 2, 3]
    selected_numbers = random.sample(set, numbers)
    selected_numbers.sort()
    return selected_numbers

def GNNs_generate_datas(Number_of_layers = 5):

    num_of_layers = Number_of_layers
    circuit_list = [] # the circuit composed of gates, such Rx, Ry and CNOT

    for layer in range(num_of_layers):
        number_of_rx = random.randint(0, 4) # the numbers of selected qubit_rx
        #number_of_rx = 4
        selected_qubit_rx = random_selection(number_of_rx)
        for qubit_rx in selected_qubit_rx:
            circuit_list.append(('rx', qubit_rx))

        number_of_ry = random.randint(0, 4)  # the numbers of selected qubit_ry
        #number_of_ry = 4
        selected_qubit_ry = random_selection(number_of_ry)
        for qubit_ry in selected_qubit_ry:
            circuit_list.append(('ry', qubit_ry))

        number_of_cnot = random.randint(0, 4)  # the numbers of selected qubit_cnot
        #number_of_cnot = 4
        selected_qubit_cnot = random_selection(number_of_cnot)
        for qubit_cnot in selected_qubit_cnot:
            circuit_list.append(('cx', qubit_cnot, (qubit_cnot + 1) % 4))

    return circuit_list

#circuits = GNNs_generate_datas(5)
#print(circuits)

