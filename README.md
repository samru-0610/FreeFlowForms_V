# Free-form flows 

## Installation

To run our experiments, install the dependencies first:

```bash
git clone https://github.com/vislearn/FFF.git
cd FFF
pip install -r requirements.txt
```

If you want to import our loss into your project, install our package using `pip`:

```bash
pip install .
```
In the last line, use `pip install -e .` if you want to edit our code.

Then you can import the package via

```python
import fff
```

## Vrushali's Experiments: 

### Experiment 1 : UNet used for reconstruction and density learning of MNIST
A lightweight U-Net was used as both encoder and decoder within the FFF framework to train on the MNIST dataset. This tests the flexibility of FFF with non-standard architectures and serves as a visual demonstration of one-step sampling and reconstruction.

Run the "train_UNet.py" file. 

**The only thing here that is borrowed from the original code is the surrogate_loss calculation.**

### Experiment 2 : Gradient Curve Analysis
We reproduced the cubic function toy example from Section 4.3 of the paper to empirically validate the claim that both the surrogate loss L_g and the inverse loss L_f-1 share the same critical points. We plot and compare gradients for both objectives across function parameterizations.

Look at the "gradient_plotting.ipynb". 
To run your own functions, you will have to hand-calculate the Lf-1 and Lg and then update the functions. 

**Nothing is borrowed from the original code**

### Everything apart from the following are the same as the original code:
1. README.md
2. train_UNet.py
3. gradient_plotting.ipynb

Note: I ran the UNet experiment over multiple ipynb files and google colab sessions, hence there are no logged plots here. But all the plots I used in my presentation are reproducible. 
