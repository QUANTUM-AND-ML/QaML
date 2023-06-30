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




## Quantum simulation
Relevant scripts and data for the paper entitled "Practical Circuit Optimization Algorithm for Quantum Simulation Based on Template Matching"

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
- Pytket >= 1.2
- Parallel computing may require NVIDIA GPUs acceleration

## Benchmarking environment
| S/N | Package |  Version  || S/N | Package |  Version  |
|:-----:| :-----: | :-----: |:-----:| :-----:| :-----: | :-----: | 
|1|Jinja2	|3.0.3	||72|pandas	|1.4.2	|
|2|Mako	|1.2.3	||73|pandocfilters	|1.5.0	|
|3|MarkupSafe	|2.1.1	||74|parso	|0.8.3	|
|4|Pillow	|9.1.0	||75|pbr	|5.9.0	|
|5|PyMetis	|2022.1|	|76|pexpect	|4.8.0	|
|6|PyYAML	|6.0	||77|pickleshare	|0.7.5	|
|7|Pygments	|2.12.0	||78|pip	|22.2.2	|
|8|Quandl	|3.7.0	||79|pkgutil-resolve-name	|1.3.10	|
|9|SQLAlchemy	|1.4.42|	|80|ply	|3.11	|
|10|Send2Trash	|1.8.0	||81|prometheus-client	|0.14.1	|
|11|alembic	|1.8.1	||82|prompt-toolkit	|3.0.29	|
|12|argon2-cffi	|21.3.0	||83|psutil	|5.9.0	|
|13|argon2-cffi-bindings	|21.2.0	||84|ptyprocess	|0.7.0	|
|14|asttokens	|2.0.5	||85|pure-eval	|0.2.2	|
|15|async-generator	|1.10	||86|pyOpenSSL	|22.1.0	|
|16|attrs	|21.4.0	||87|pycparser	|2.21	|
|17|backcall	|0.2.0	||88|pydot	|1.4.2	|
|18|backports.functools-lru-cache	|1.6.4|	|89|pylatexenc	|2.10	|
|19|beautifulsoup4	|4.11.1	||90|pyparsing	|3.0.9	|
|20|bleach	|5.0.0	||91|pyrsistent	|0.18.1	|
|21|certifi	|2022.9.24	||92|pyscf	|2.0.1	|
|22|certipy	|0.1.3	||93|python-constraint	|1.4.0	|
|23|cffi	|1.15.1	||94|python-dateutil	|2.8.2	|
|24|charset-normalizer	|2.0.12	||95|python-json-logger	|2.0.4	|
|25|cryptography	|38.0.1	||96|pytket	|1.2.2	|
|26|cycler	|0.11.0	||97|pytket-qiskit	|0.24.0	|
|27|debugpy	|1.6.0	||98|pytz	|2022.1	|
|28|decorator	|5.1.1	||99|pyzmq	|23.2.0	|
|29|defusedxml	|0.7.1	||100|qiskit	|0.36.1	|
|30|dill	|0.3.4	||101|qiskit-aer	|0.10.4	|
|31|dlx	|1.0.4	||102|qiskit-aqua	|0.9.5	|
|32|docplex	|2.23.222	||103|qiskit-ibmq-provider	|0.19.1	|
|33|entrypoints	|0.4	||104|qiskit-ignis	|0.7.0	|
|34|executing	|0.8.3	||105|qiskit-nature	|0.3.2	|
|35|fastdtw	|0.3.4	||106|qiskit-terra	|0.20.1	|
|36|fastjsonschema	|2.16.2	||107|requests	|2.27.1	|
|37|fonttools	|4.33.3	||108|requests-ntlm	|1.1.0	|
|38|graphviz	|0.20	||109|retworkx	|0.11.0	|
|39|greenlet	|1.1.3.post0	||110|ruamel.yaml	|0.17.21	|
|40|h5py	|3.2.1	||111|ruamel.yaml.clib	|0.2.6	|
|41|idna	|3.3	||112|scikit-learn	|1.1.1	|
|42|importlib-resources	|5.7.1	||113|scipy	|1.8.0	|
|43|inflection	|0.5.1	||114|seaborn	|0.11.2	|
|44|ipykernel	|6.15.2	||115|setuptools	|63.4.1	|
|45|ipython	|8.4.0	||116|six	|1.16.0	|
|46|ipython-genutils	|0.2.0	||117|soupsieve	|2.3.2.post1	|
|47|ipywidgets	|7.7.0	||118|stack-data	|0.2.0	|
|48|jedi	|0.18.1	||119|stevedore	|3.5.0	|
|49|joblib	|1.1.0	||120|symengine	|0.9.2	|
|50|jsonschema	|4.16.0	||121|sympy	|1.10.1	|
|51|kahypar	|1.1.7	||122|tensornetwork	|0.4.6	|
|52|kiwisolver	|1.4.2	||123|terminado	|0.15.0	|
|53|lark-parser	|0.12.0	||124|testpath	|0.6.0	|
|54|matplotlib	|3.5.2	||125|threadpoolctl	|3.1.0	|
|55|matplotlib-inline	|0.1.6	||126|tinycss2	|1.1.1	|
|56|mistune	|0.8.4	||127|tornado	|6.2	|
|57|more-itertools	|8.13.0	||128|tqdm	|4.64.0	|
|58|mpmath	|1.2.1	||129|traitlets	|5.2.1.post0	|
|59|multitasking	|0.0.10	||130|tweedledum	|1.1.1	|
|60|nbclient	|0.6.3	||131|types-pkg-resources	|0.1.3	|
|61|nbconvert	|6.5.0	||132|typing-extensions	|4.3.0	|
|62|nbformat	|5.5.0	||133|urllib3	|1.26.9	|
|63|nest-asyncio	|1.5.5	||134|wcwidth	|0.2.5	|
|64|networkx	|2.8.2	||135|webencodings	|0.5.1	|
|65|notebook	|6.4.12	||136|websocket-client	|1.3.2	|
|66|ntlm-auth	|1.5.0	||137|websockets	|10.3	|
|67|numpy	|1.22.3	||138|wheel	|0.37.1	|
|68|oauthlib	|3.2.2	||139|widgetsnbextension	|3.6.0	|
|69|opt-einsum	|3.3.0	||140|yfinance	|0.1.70	|
|70|packaging	|21.3	||141|zipp	|3.8.0	|
|71|pamela	|1.0.0	|
