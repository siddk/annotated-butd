# Annotated Bottom-Up Top-Down Attention for Visual Question Answering

Companion Repository to the Annotated Bottom-Up Top-Down Attention (BUTD) for VQA Blog Post, written by
Siddharth Karamcheti. 

## About
The repository is factored into two branches:
 - The **[Modular Branch]** contains a fully factored version of the BUTD 
   codebase, broken apart into different modules for pre-processing, model creation, and training, for
   VQA2, GQA, and NLVR2.
   
 - The **[Streamlined Branch]** contains a single-file annotated version of the BUTD codebase only for the NLVR2 task.
 
## Repository Overview
The repository contains the following components:


## Quickstart


## Start-Up (from Scratch)

Use these commands if you don't use Conda/don't trust the environment.yml for whatever reason. The following contains
step-by-step instructions for creating a new Conda Environment and installing the necessary dependencies.

```bash
# Create & Activate Conda Environment
conda create --name annotated-butd python=3.7
conda activate annotated-butd

# Mac OS/Linux (if using GPU, make sure CUDA already installed)
conda install pytorch torchvision -c pytorch
conda install ipython jupyter 
pip install pytorch-lightning typed-argument-parser h5py opencv-python matplotlib
``` 
   
