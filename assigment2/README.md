# DeepXplore: Systematic DNN testing  (SOSP'17)
This repository implements “difference-inducing input generation” based on the SOSP ’17 paper [DeepXplore: Automated Whitebox Testing of Deep Learning Systems](http://www.cs.columbia.edu/~suman/docs/deepxplore.pdf).

## Prerequisite
### Python
The code should be run using python 3.9.21

### **Install dependencies**
```bash
pip install -r requirements.txt
```

## File structure
+ **MNIST** - Difference-inducing input generation for MNIST
+ **ImageNet** - Difference-inducing input generation for ImageNet
+ **CIFAR** - Difference-inducing input generation for CIFAR-10
+ **PDF** - PDF document reader model testing
+ **mimicus** - Install from [here](https://github.com/srndic/mimicus).

# To run
In every directory
```bash
python CIFAR/gen_diff.py occl 0.5 0.3 0.01 10 20 0.001 -t 1 -sp 5 5 -occl_size 8 8
python MNIST/gen_diff.py occl 0.5 0.3 0.01 10 20 0.001 -t 1 -sp 5 5 -occl_size 8 8
python ImageNet/gen_diff.py occl 0.5 0.3 0.01 10 20 0.001 -t 1 -sp 5 5 -occl_size 8 8
python PDF/gen_diff.py 0.5 0.3 0.01 10 20 0.001 -t 0
```

# Note
The trained weights are provided in each directory (if required).
Drebin's weights are not part of this repo as they are too large to be hosted on GitHub. Download from [here](https://drive.google.com/drive/folders/0B4otJeEcboCaQzFpYkJwb2h3WG8?resourcekey=0-ns2toseJWe6qVS0nOl6rnw&usp=sharing) and put them in ./Drebin/.
