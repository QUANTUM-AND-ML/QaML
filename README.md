<p align="center">
<img src="Q&ML.png" alt="Q&ML Logo" width="600">
</p>

<h2><p align="center">A PyThon Library for Quantum Computation and Machine Learning</p></h2>
<h3><p align="center">Updated, Scalable, Easy Implement, Easy Reading and Comprehension</p></h3>


<p align="center">
    <a href="https://github.com/QUANTUM-AND-ML/QaML/blob/main/LICENSE">
        <img alt="MIT License" src="https://img.shields.io/github/license/QUANTUM-AND-ML/QUANTUM-QuantumSimulation">
    </a>
    <a href="https://www.python.org/downloads/release/python-3813/">
        <img alt="Version" src="https://img.shields.io/badge/Python-3.8-orange">
    </a>
    <a href="https://github.com/search?q=repo%3AQUANTUM-AND-ML%2FQaML++language%3APython&type=code">
        <img alt="Language" src="https://img.shields.io/github/languages/top/QUANTUM-AND-ML/QUANTUM-QuantumSimulation">
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

s


## Quantum & Machine Learning
Relevant scripts and data for the paper entitled "Output Estimation of Quantum Circuits based on Graph Neural Network"

## Table of contents
* [**Previous work**](#Previous-work)
* [**Python scripts**](#Python-scripts)
* [**Dependencies**](#Dependencies)
* [**Benchmarking environment**](#Benchmarking-environment)

## Previous work
Our work in this paper builds on previous quantum simulation work. Our previous submission is called "[**Greedy algorithm based circuit optimization for near-term quantum simulation**](https://iopscience.iop.org/article/10.1088/2058-9565/ac796b)". In previous work, we develop a hardware-agnostic circuit optimization algorithm to reduce the overall circuit cost for Hamiltonian simulation problems. Our method employ a novel sub-circuit synthesis in intermediate representation and propose a greedy ordering scheme for gate cancellation to minimize the gate count and circuit depth.
## Python scripts
Everyone can change the value of the parameter "**Hamiltonian =**" in the **main.py** file to compare the results of different optimizers.  

**TemplateMatching.py** file includes functions for **Quantum circuit optimization** and **Template circuit preparation**：

>**Quantum circuit optimization**：
>
>Including two consecutive CNOTs can be eliminated, two identical rotational gates can be fused, etc.
>
>**Template circuit preparation**：
>
>As an example, if all two-local pauli operators appear between the same qubits, they are arranged in the order **XY, XZ, ZX, YX, YY, YZ, ZY, XX, ZZ**. And each two-local pauli operator is transformed into a **suitable equivalent circuit** ( proved by the rules of [**ZX-calculus**](https://zxcalculus.com/)) , and then a template is obtained after circuit optimization.

## Dependencies
- 3.9 >= Python >= 3.7 (Python 3.10 may have the `concurrent` package issue for Qiskit)
- Qiskit >= 0.36.1
- PyTorch >= 1.8.0
- Parallel computing may require NVIDIA GPUs acceleration

## Benchmarking environment

