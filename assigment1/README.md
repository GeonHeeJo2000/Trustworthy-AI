## Assignment 1: Adversarial Attacks on MNIST and CIFAR-10

In this assignment, we:
- Implement and compare **untargeted vs targeted** adversarial attacks
- Use **FGSM (Fast Gradient Sign Method)** and **PGD (Projected Gradient Descent)**
- Apply these attacks to CNNs trained on **MNIST** and **CIFAR-10**
- Evaluate both **accuracy** and **attack success rate**
- Visualize perturbed inputs and compare their model predictions

---

## 🔧 Repository Structure

```
Trustworthy-AI
└── assignment1/
    ├── test.py          # Script for training and running attacks
    ├── test.ipynb       # Jupyter notebook version for interactive experiments 
    ├── data/            # Contains manually downloaded datasets (e.g., MNIST, CIFAR)
    │   └── MNIST/
    │       └── raw/     # Raw MNIST dataset files should be placed here
    │                    # Download from: http://yann.lecun.com/exdb/mnist/
    └── requirements.txt # Python dependencies for this assignment
```

---

## 📦 Requirements

- Python 3.11.0 required
```bash
pip install -r requirements.txt
```

---

## 📂 Dataset Setup

### 1. MNIST
- These will be automatically extracted and processed by `torchvision.datasets.MNIST(..., download=True)`.
- Due to occasional **network errors**, MNIST cannot always be downloaded automatically.
- If that happens, you can manually download the files and place them under `./data/MNIST/raw/`:

```plaintext
https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz  
https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz  
https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz  
https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz  
```

### 2. CIFAR-10
- CIFAR-10 **downloads automatically** when using `torchvision.datasets.CIFAR10(..., download=True)`

---

## 🚀 How to Run

```bash
python test.py
```

or open:

```bash
test.ipynb
```
📌 *You can freely switch between MNIST and CIFAR by changing the model and dataset loaders in `test.py` or `test.ipynb`*
