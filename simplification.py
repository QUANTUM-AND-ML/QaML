import operator

# Judge the specific details of the list marks and connections of the neighboring quanta in the Quantum circuit
def neighbor(ansatz,order):
    '''
    :param ansatz: Input Quantum circuit structure, list format： Example: [('rx', 0), ('rz', 1), ('cx', 0, 1)];
    :param order: List mark of the Quantum circuit structure where the quantum gate is located; Example: the first quantum gate is marked as 0 in the list
    :return:Control_Bit: The initial value is [-1, -1], and single bit gates only consider this. The first bit in the list is the sequence number of the adjacent gates in front, 
    the two bit gate is the control bit corresponding to the previous sequence number, and the second bit in the list is the sequence number of the adjacent gates in the back. The two bits are similar,
            Controlled_Bit: Initially [-1, -1], the two bit gate controlled bits correspond to the sequence numbers of adjacent gates. 
            The first digit in the list is the sequence number of the previous adjacent list, and the second digit in the list is the sequence number of the subsequent adjacent gates,
            Quantum_Gate_Marking: The initial value is [0, 0, 0, 0], a list of four bits. The first two represent the types of neighboring gates before and after the control bit, 
            1 represents a single bit gate, 2 represents the control bit of a two bit gate, 
            3 represents the controlled bit of a two bit gate, and the last two represent the types of neighboring gates before and after the controlled bit;
    '''
    Control_Bit = [-1,-1]
    Controlled_Bit = [-1,-1]
    Quantum_Gate_Marking = [0,0,0,0]

    # Determine the neighbor numbers before and after Rx and Rz
    if ansatz[order][0] == 'h' or ansatz[order][0] == 't':
        #Find previous neighbor number
        if order == 0:
            Control_Bit[0] = -1
            Controlled_Bit[0] = -1
        else:
            for i in range(order-1,-1,-1):
                if ansatz[i][0] == 'h' or ansatz[i][0] == 't':
                    if ansatz[i][1] == ansatz[order][1]:
                        Control_Bit[0] = i
                        Quantum_Gate_Marking[0] = 1
                        break
                elif ansatz[i][0] == 'cx':
                    if ansatz[i][1] == ansatz[order][1]:
                        Control_Bit[0] = i
                        Quantum_Gate_Marking[0] = 2
                        break
                    elif ansatz[i][2] == ansatz[order][1]:
                        Control_Bit[0] = i
                        Quantum_Gate_Marking[0] = 3
                        break
                else:
                    print('Unknown quantum gate appears')
                    break
        # Find the next neighbor number
        if order == len(ansatz):
            Control_Bit[1] = -1
            Controlled_Bit[1] = -1
        else:
            for i in range(order+1,len(ansatz),1):
                if ansatz[i][0] == 'h' or ansatz[i][0] == 't':
                    if ansatz[i][1] == ansatz[order][1]:
                        Control_Bit[1] = i
                        Quantum_Gate_Marking[1] = 1
                        break
                elif ansatz[i][0] == 'cx':
                    if ansatz[i][1] == ansatz[order][1]:
                        Control_Bit[1] = i
                        Quantum_Gate_Marking[1] = 2
                        break
                    elif ansatz[i][2] == ansatz[order][1]:
                        Control_Bit[1] = i
                        Quantum_Gate_Marking[1] = 3
                        break
                else:
                    print('Unknown quantum gate appears')
                    break

    # Determine the neighbor numbers before and after Cx
    elif ansatz[order][0] == 'cx':
        # Find the neighbor number of the previous control bit
        if order == 0:
            Control_Bit[0] = -1
            Controlled_Bit[0] = -1
        else:
            for i in range(order-1,-1,-1):
                if ansatz[i][0] == 'h' or ansatz[i][0] == 't':
                    if ansatz[i][1] == ansatz[order][1]:
                        Control_Bit[0] = i
                        Quantum_Gate_Marking[0] = 1
                        break
                if ansatz[i][0] == 'cx':
                    if ansatz[i][1] == ansatz[order][1]:
                        Control_Bit[0] = i
                        Quantum_Gate_Marking[0] = 2
                        break
                    elif ansatz[i][2] == ansatz[order][1]:
                        Control_Bit[0] = i
                        Quantum_Gate_Marking[0] = 3
                        break
        # Find the neighbor number for the subsequent control bit
        if order == len(ansatz):
            Control_Bit[1] = -1
            Controlled_Bit[1] = -1
        else:
            for i in range(order+1,len(ansatz),1):
                if ansatz[i][0] == 'h' or ansatz[i][0] == 't':
                    if ansatz[i][1] == ansatz[order][1]:
                        Control_Bit[1] = i
                        Quantum_Gate_Marking[1] = 1
                        break
                if ansatz[i][0] == 'cx':
                    if ansatz[i][1] == ansatz[order][1]:
                        Control_Bit[1] = i
                        Quantum_Gate_Marking[1] = 2
                        break
                    elif ansatz[i][2] == ansatz[order][1]:
                        Control_Bit[1] = i
                        Quantum_Gate_Marking[1] = 3
                        break
        # Find the neighbor number of the previously controlled bit
        if order == 0:
            Control_Bit[0] = -1
            Controlled_Bit[0] = -1
        else:
            for i in range(order-1,-1,-1):
                if ansatz[i][0] == 'h' or ansatz[i][0] == 't':
                    if ansatz[i][1] == ansatz[order][2]:
                        Controlled_Bit[0] = i
                        Quantum_Gate_Marking[2] = 1
                        break
                if ansatz[i][0] == 'cx':
                    if ansatz[i][1] == ansatz[order][2]:
                        Controlled_Bit[0] = i
                        Quantum_Gate_Marking[2] = 2
                        break
                    elif ansatz[i][2] == ansatz[order][2]:
                        Controlled_Bit[0] = i
                        Quantum_Gate_Marking[2] = 3
                        break
        # Find the neighbor number of the next controlled bit
        if order == len(ansatz):
            Control_Bit[1] = -1
            Controlled_Bit[1] = -1
        else:
            for i in range(order+1,len(ansatz),1):
                if ansatz[i][0] == 'h' or ansatz[i][0] == 't':
                    if ansatz[i][1] == ansatz[order][2]:
                        Controlled_Bit[1] = i
                        Quantum_Gate_Marking[3] = 1
                        break
                if ansatz[i][0] == 'cx':
                    if ansatz[i][1] == ansatz[order][2]:
                        Controlled_Bit[1] = i
                        Quantum_Gate_Marking[3] = 2
                        break
                    elif ansatz[i][2] == ansatz[order][2]:
                        Controlled_Bit[1] = i
                        Quantum_Gate_Marking[3] = 3
                        break
    else:
        print('Looking for an unknown quantum gate')
    return Control_Bit,Controlled_Bit,Quantum_Gate_Marking

# Search for the subscript in the fixed list mark and parameter list in the Quantum circuit. The subscript of the CNOT gate is the subscript of the Revolving door parameter in front of it
def Search_parameters(ansatz, order):
    new_order = 0
    for i in range(0,order,1):
        if ansatz[i][0] == 'h' or ansatz[i][0] == 't':
            new_order = new_order + 1
    return new_order

# Delete initial CNOT gate
def rule_1(ansatz):
    '''
    :param ansatz: Input Quantum circuit structure, list format; Example: [('rx', 0), ('rz', 1), ('cx', 0, 1)];
    :return: Quantum circuit structure after using rule 1;
    '''
    new_ansatz = list(ansatz)
    Count = len(new_ansatz)
    i = 0
    label = 0 # Indicates that the rule does not use.
    while Count > 0:
        if new_ansatz[i][0] == 'cx':
            Control_Bit, Controlled_Bit, Quantum_Gate_Marking = neighbor(new_ansatz, i)
            if Control_Bit[0] == -1:
                new_ansatz.pop(i)
                label = 1
                i = i - 1
        i = i + 1
        Count = Count - 1
    return new_ansatz, label

# Delete the initial t-gate
def rule_2(ansatz):
    new_ansatz = list(ansatz)
    Count = len(new_ansatz)
    label = 0  # Indicates that the rule does not use.
    i = 0
    while Count > 0:
        if new_ansatz[i][0] == 't':
            Control_Bit, Controlled_Bit, Quantum_Gate_Marking = neighbor(new_ansatz, i)
            if Control_Bit[0] == -1:
                new_ansatz.pop(i)
                label = 1
                i = i - 1
        i = i + 1
        Count = Count - 1
    return new_ansatz, label

# Delete two duplicate CNOT gates
def rule_3(ansatz):
    new_ansatz = list(ansatz)
    Count = len(new_ansatz)
    i = 0
    label = 0  # Indicates that the rule does not use.
    while Count > 0:
        if new_ansatz[i][0] == 'cx':
            Control_Bit, Controlled_Bit, Quantum_Gate_Marking = neighbor(new_ansatz, i)
            if Control_Bit[1] == Controlled_Bit[1] and Control_Bit[1] != -1 and Quantum_Gate_Marking[1] == 2:
                new_ansatz.pop(Control_Bit[1])
                new_ansatz.pop(i)
                label = 1
                i = i - 1
                Count = Count - 1
        i = i + 1
        Count = Count - 1
    return new_ansatz, label

# Delete two identical H-gates
def rule_4(ansatz):
    new_ansatz = list(ansatz)
    Count = len(new_ansatz)
    i = 0
    label = 0  # Indicates that the rule does not use.
    while Count > 0:
        if new_ansatz[i][0] == 'h':
            Control_Bit, Controlled_Bit, Quantum_Gate_Marking = neighbor(new_ansatz, i)
            if Control_Bit[1] != -1 and new_ansatz[Control_Bit[1]][0] == 'h':
                new_ansatz.pop(Control_Bit[1])
                new_ansatz.pop(i)
                label = 1
                i = i - 1
                Count = Count - 1
        i = i + 1
        Count = Count - 1
    return new_ansatz, label

# Delete the last t-gate
def rule_5(ansatz):
    new_ansatz = list(ansatz)
    Count = len(new_ansatz)
    label = 0  # Indicates that the rule does not use.
    i = 0
    while Count > 0:
        if new_ansatz[i][0] == 't':
            Control_Bit, Controlled_Bit, Quantum_Gate_Marking = neighbor(new_ansatz, i)
            if Control_Bit[1] == -1:
                new_ansatz.pop(i)
                label = 1
                i = i - 1
        i = i + 1
        Count = Count - 1
    return new_ansatz, label

# Single-qubit gate t moves to the right
def rule_Tshift_right(ansatz):
    new_ansatz = list(ansatz)
    label = 0  # Indicates that the rule does not use.
    for i in range(0,len(new_ansatz),1):
        if new_ansatz[i][0] == 't':
            Control_Bit, Controlled_Bit, Quantum_Gate_Marking = neighbor(new_ansatz, i)
            if Control_Bit[1] != -1 and  Quantum_Gate_Marking[1] == 2:
                new_ansatz.insert(Control_Bit[1] + 1,new_ansatz[i])
                new_ansatz.pop(i)
                label = 1
    return new_ansatz, label

# Simplify quantum circuits according to rules
def simplification(ansatz):
    new_ansatz = ansatz
    sum_label = 1
    while sum_label > 0:
        sum_label = 0
        new_ansatz, label = rule_1(new_ansatz)
        sum_label += label
        new_ansatz, label = rule_2(new_ansatz)
        sum_label += label
        new_ansatz, label = rule_3(new_ansatz)
        sum_label += label
        new_ansatz, label = rule_4(new_ansatz)
        sum_label += label
        new_ansatz, label = rule_5(new_ansatz)
        sum_label += label
    label = 1
    while label > 0:
        new_ansatz, b = rule_Tshift_right(new_ansatz)
        label = b
        #print(label)
    sum_label = 1
    while sum_label > 0:
        sum_label = 0
        new_ansatz, label = rule_1(new_ansatz)
        sum_label += label
        new_ansatz, label = rule_2(new_ansatz)
        sum_label += label
        new_ansatz, label = rule_3(new_ansatz)
        sum_label += label
        new_ansatz, label = rule_4(new_ansatz)
        sum_label += label
        new_ansatz, label = rule_5(new_ansatz)
        sum_label += label

    return new_ansatz
