# Annotated Bottom-Up Top-Down Attention for Visual Question Answering -- [gqa-ipynb]

Companion Repository to the Annotated Bottom-Up Top-Down Attention (BUTD) for VQA Blog Post. 

The goal of this repository is to facilitate VQA research by making accessible a set of strong baseline VQA models 
(that don't rely on expensive pre-training!) that are easily hackable. 

Furthermore, by including pre-processing and training code for 3 common VQA datasets 
(GQA, NLVR-2, and VQA-2), we hope to make it easier to perform comprehensive cross-dataset evaluations in the future.

## About
The repository is factored into multiple branches:
 - The **[Modular Branch]** contains a fully factored version of the BUTD 
   codebase, broken apart into different modules for pre-processing, model creation, and training, for
   VQA2, GQA, and NLVR2. Use this branch for most research/development purposes.
   
 - The **[Dataset-ipynb Branches]** contain a single-file annotated IPython Notebook of the BUTD codebase 
   for each of the various VQA tasks. Use these branches to slowly step through the code (to better understand 
   pre-processing intricacies, model design choices, etc.)
 
## Repository Overview
This branch (**[GQA-ipynb]**) contains the following components:

- `gqa.ipynb` - Standalone IPython Notebook that steps through data download, preprocessing, model definition, and 
                training.

## Quickstart

Use these commands to quickly get set up with this repository, and start running experiments on GQA, NLVR-2, and VQA-2.

```bash 
# Clone the Repository
git clone https://github.com/siddk/annotated-butd.git
cd annotated-butd
git checkout gqa-ipynb

# Launch Jupyter Notebook (Python 3 Kernel)
jupyter notebook gqa.ipynb
```