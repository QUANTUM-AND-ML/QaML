<p align="center">
<img src="figures/Q&ML.png" alt="Q&ML Logo" width="600">
</p>

<h2><p align="center">A PyThon Library for Quantum Computation and Machine Learning</p></h2>
<h3><p align="center">Updated, Scalable, Easy Implement, Easy Reading and Comprehension</p></h3>


<p align="center">
    <a href="https://github.com/QUANTUM-AND-ML/QaML/blob/main/LICENSE">
        <img alt="MIT License" src="https://img.shields.io/github/license/QUANTUM-AND-ML/QaML">
    </a>
    <a href="https://www.python.org/downloads/release/python-3813/">
        <img alt="Version" src="https://img.shields.io/badge/Python-3.8-orange">
    </a>
    <a href="https://github.com/search?q=repo%3AQUANTUM-AND-ML%2FQaML++language%3APython&type=code">
        <img alt="Language" src="https://img.shields.io/github/languages/top/QUANTUM-AND-ML/QaML">
    </a>
   <a href="https://github.com/QUANTUM-AND-ML/QaML/activity">
        <img alt="Activity" src="https://img.shields.io/github/last-commit/QUANTUM-AND-ML/QUANTUM-QuantumSimulation">
    </a>
       <a href="https://www.nsfc.gov.cn/english/site_1/index.html">
        <img alt="Fund" src="https://img.shields.io/badge/supported%20by-NSFC-green">
    </a>
    <a href="https://twitter.com/FindOne0258">
        <img alt="twitter" src="https://img.shields.io/badge/twitter-chat-2eb67d.svg?logo=twitter">
    </a>


</p>
<br />



## Quantum & Machine Learning
Relevant scripts and data for the paper entitled "Output Prediction of Quantum Circuits based on Graph Neural Networks"

## Table of contents
* [**Main work**](#Main-work)
* [**Results display**](#Results-display)
* [**Python scripts**](#Python-scripts)
* [**Dependencies**](#Dependencies)

## Main work
This paper proposes a Graph Neural Networks (GNNs)-based framework to predict the output expectation values of quantum circuits under noisy and noiseless conditions and compare the performance of different parameterized quantum circuits (PQCs).
We begin by constructing datasets under noisy and noiseless conditions using a non-parameterized quantum gate set to predict circuit expectation values. The node feature vectors for GNNs are specifically designed to include noise information. In our simulations, we compare the prediction performance of GNNs in both noisy and noiseless conditions, against Convolutional Neural Networks (CNNs) on the same dataset and their qubit scalability. GNNs demonstrate superior prediction accuracy across diverse conditions. Subsequently, we utilize the parameterized quantum gate set to construct noisy PQCs and compute the ground state energy of hydrogen molecules using the Variational Quantum Eigensolver (VQE).
We propose two schemes: the Indirect Comparison scheme, which involves directly predicting the ground state energy and subsequently comparing circuit performances, and the Direct Comparison scheme, which directly predicts the relative performance of the two circuits. Simulation results indicate that the Direct Comparison scheme significantly outperforms the Indirect Comparison scheme by an average of 36.2\% on the same dataset, providing a new and effective perspective for using GNNs to predict the overall properties of PQCs, specifically by focusing on their performance differences.

<p align="center">
<img src="figures/Figure_1.png" alt="Figure 1" width="800">
</p>

**Figure 1.** The framework for expectation values prediction. **a)** Generate random quantum circuits. **b)** Transform the random quantum circuits into graph structures. **c)** Incorporate noise information into the graph nodes of the quantum circuits, train GNNs, and predict the expectation values of quantum circuits.

<p align="center">
<img src="figures/Figure_2.png" alt="Figure 2" width="760">
</p>

**Figure 2.** The framework of circuit performance comparison prediction. Firstly, convert the two quantum circuits into one graph, where each node feature vector includes information about the noise. Input the graph into the GNNs to directly predict the performance comparison of the two circuits.


## Results display


<p align="center">
<img src="figures/Figure_3.png" alt="Figure 3" width="560">
</p>

**Figure 3.** Single-qubit expectation values prediction by GNNs under noisy and noiseless conditions.

<p align="center">
<img src="figures/Figure_4.png" alt="Figure 4" width="560">
</p>

**Figure 4.** Prediction accuracy of Indirect Comparison and Direct Comparison schemes for parametrized quantum circuit performance.

## Python scripts
Here is the **brief introduction** to each python file for better understanding and usage:

* "main.py" primarily includes dataset splitting and the **training** and **testing** of the neural network.
* "Dataset.py" mainly includes the generation and saving of a large amount of datasets.
* "CNN_generate_datas.py" mainly implements the generation of random quantum circuits used for training **Convolutional Neural Networks** (*CNNs*), as mentioned in the paper "[**Supervised learning of random quantum circuits via scalable neural networks**](https://iopscience.iop.org/article/10.1088/2058-9565/acc4e2)."
* "GNN_generate_datas.py" mainly implements the generation of random quantum circuits used for training **Graph Neural Networks** (*GNNs*) to compute the ground-state energy of hydrogen molecules.
* "circuit_to_graph.py" primarily defines a class that allows **the conversion of quantum circuits into graphs**.
* "expectation_and_ground_state_energy_calculation.py" mainly defines functions for calculating the expectation values of quantum circuits and the ground state energy of the hydrogen molecule.
* "CNNs.py" and "GNNs.py" respectively contain the architectures of the constructed **Convolutional Neural Networks** (*CNNs*) and **Graph Neural Networks** (*GNNs*).
* "simplification.py" mainly contains some rules for **simplifying quantum circuits**.

## Dependencies
- 3.9 >= Python >= 3.7 (Python 3.10 may have the `concurrent` package issue for Qiskit)
- Qiskit >= 0.36.1
- PyTorch >= 1.8.0
- Parallel computing may require NVIDIA GPUs acceleration
