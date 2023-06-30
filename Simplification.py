#import pennylane as qml
#from pennylane import numpy as np
import operator


#判断量子线路中量子们相邻量子们的列表标记和连接的具体细节
def neighbor(ansatz,order):
    '''
    :param ansatz:输入的量子线路结构，列表格式; Example: [('rx', 0), ('rz', 1), ('cx', 0, 1)];
    :param order: 所查找量子门所处量子线路结构的列表标记; Example:第一个量子门，所处列表标记为0
    :return:Control_Bit:初始为[-1,-1],单比特门只考虑这个，列表的第一位是前面相邻门的序号，两比特门是控制位对应前面序号，列表第二位是后面相邻门序号，两比特类似,
            Controlled_Bit:初始为[-1，-1]，两比特门受控位对应相邻门的序号，列表第一位是前面相邻列表的序号，列表第二位是后面相邻门序号,
            Quantum_Gate_Marking:初始为[0，0，0，0]，共四位的列表，前两个代表控制比特前后邻居门的类型，1表示单比特门，2表示两比特门的控制位，3表示两比特门的受控位，后两个表示受控比特的前后邻居门的类型;
    '''
    Control_Bit = [-1,-1]
    Controlled_Bit = [-1,-1]
    Quantum_Gate_Marking = [0,0,0,0]

    #判断Rx,Rz前后的邻居编号
    if ansatz[order][0] == 'h' or ansatz[order][0] == 't':
        #寻找前面邻居编号
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
                    print('出现未知量子门')
                    break
        #寻找后面邻居编号
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
                    print('出现未知量子门')
                    break

    #判断Cx前后的邻居编号
    elif ansatz[order][0] == 'cx':
        #寻找前面控制比特的邻居编号
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
        #寻找后面控制比特的邻居编号
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
        #寻找前面受控制比特的邻居编号
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
        #寻找后面受控制比特的邻居编号
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
        print('查找的是未知的量子门')
    return Control_Bit,Controlled_Bit,Quantum_Gate_Marking

#搜寻量子线路中固定列表标记再参数列表中的下标,CNOT门的下标是其前面旋转门参数的下标
def Search_parameters(ansatz, order):
    new_order = 0
    for i in range(0,order,1):
        if ansatz[i][0] == 'h' or ansatz[i][0] == 't':
            new_order = new_order + 1
    return new_order


#删除初始的CNOT门
def rule_1(ansatz):
    '''
    :param ansatz: 输入的量子线路结构，列表格式; Example: [('rx', 0), ('rz', 1), ('cx', 0, 1)];
    :return: 使用规则一后的量子线路结构;
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
    #print('使用规则1次数:',len(ansatz) - len(new_ansatz))
    return new_ansatz, label

#删除初始的t门
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
    #print('使用规则2次数:',len(ansatz) - len(new_ansatz))
    return new_ansatz, label

#删除两个重复的CNOT门
def rule_3(ansatz):
    '''
    :param ansatz: 输入的量子线路结构，列表格式; Example: [('rx', 0), ('rz', 1), ('cx', 0, 1)];
    :return: 使用规则一后的量子线路结构;
    '''
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
    #print('使用规则3次数:',len(ansatz) - len(new_ansatz))
    return new_ansatz, label

#删除两个相同的H门
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
    #print('使用规则4次数:',len(ansatz) - len(new_ansatz))
    return new_ansatz, label

#删除最后的t门
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
    #print('使用规则5次数:',len(ansatz) - len(new_ansatz))
    return new_ansatz, label

#单比特门t向右移动
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

#按照规则，化简量子线路
def simplification(ansatz):
    #label = 1
    #while label > 0:
        #new_ansatz, label = rule_Tshift_left(ansatz)
        #print(label)
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


'''Example: [('ry', 0), ('rz', 1), ('cx', 0, 1)];'''
'''
lst= [('h', 0), ('h', 1), ('h', 2), ('t', 2), ('cx', 2, 1)]
print(lst)
a, b = rule_Tshift_right(lst)
print(a,b)
'''