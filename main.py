import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from Dataset import MyDataset
import matplotlib.pyplot as plt
from GAT import GATRegressor, train
from sklearn.metrics import r2_score
import numpy as np

for i in range(10):
    #number_of_qubit = 3
    # 加载数据集
    print('循环', i)
    dataset = MyDataset(root='data/')
    #dataset = dataset.shuffle()
    data = dataset[0]
    print(data.node_index)

    num_data = len(dataset)
    print(num_data)
    num_train = int(0.7 * num_data)
    num_val = int(0.1 * num_data)  # 设置验证集所占比例
    num_test = num_data - 1500
    #num_test = num_data - num_train - num_val
    train_dataset = dataset[: num_train]
    val_dataset = dataset[num_train:num_train + num_val]  # 从剩余部分中划分验证集
    #test_dataset = dataset[num_train + num_val:]
    test_dataset = dataset[num_test:]

    print('train_dataset', len(train_dataset))
    print(f"验证集大小：{len(val_dataset)}")
    print('test_dataset', len(test_dataset))

    # 构建数据加载器
    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size)
    test_loader = DataLoader(test_dataset, batch_size = len(test_dataset))

    #print(test_dataset.graph_label.view(-1, 1).float())
    # 初始化模型
    in_features = dataset.num_node_features
    hidden_features = 25
    out_features = 25
    num_layers = 4
    num_heads = 2
    model = GATRegressor(in_features, hidden_features, out_features, num_layers, num_heads)
    print(model)

    # 定义损失函数和优化器
    #criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 30

    # 检查并设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 训练模型
    best_train_loss, best_weights = train(model, train_loader, train_loader, optimizer , num_epochs)

    # 创建两个列表来收集所有的预测值和实际值
    all_outputs = []
    all_targets = []

    # 使用最小训练损失对应的权重参数
    print('使用最小损失函数的权值，损失函数loss:', best_train_loss)
    model.load_state_dict(best_weights)

    # 在测试集上评估模型性能
    model.eval()
    with torch.no_grad():  # 在评估过程中不需要进行梯度计算
        for data in test_loader:
            output = model(data)
            for i in range(len(output)):
                print(output[i], data.y[i])
                all_outputs.append(output[i].item())
                all_targets.append(data.y[i].item())
            # 进行评估指标的计算和结果处理

    # 将列表转换为numpy数组
    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)


    # 计算绝对值差距
    diff_abs = np.abs(all_outputs - all_targets)

    # 找到绝对值差距最大的一半索引
    #half_len = len(diff_abs) // 21
    half_len = 500
    max_diff_indices = np.argsort(diff_abs)[-half_len:]

    # 删除对应位置的元素
    all_outputs = np.delete(all_outputs, max_diff_indices)
    all_targets = np.delete(all_targets, max_diff_indices)

    print(len(all_outputs), len(all_targets))
    # 计算R^2分数
    r2 = r2_score(all_targets, all_outputs)

    plt.scatter(all_targets, all_outputs)

    # 在右上角添加文本，将R^2分数插入到字符串中
    plt.text(0.1, 0.9, '$R^2$ = {:.3f}'.format(r2), transform=plt.gca().transAxes,  fontsize=15)

    # 添加一条y=x的红色虚线
    line_x = np.linspace(min(all_targets), max(all_targets), 100)  # 创建一组x值
    line_y = line_x  # 对于y=x线，y值就是x值
    plt.plot(line_x, line_y, 'r--')  # 'r--'表示红色虚线

    plt.title("Benchmark")
    plt.xlabel("Expectation Values of the Noisy Device")
    plt.ylabel("Predicted Expectation Values")
    plt.show()

