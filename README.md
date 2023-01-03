CancerGene
==============================

This is a model trained to recognize two types of cancer:
acute myeloid leukemia (AML) and acute lymphoblastic leukemia (ALL).

This project demonstrates the process of handling DNA data, using PCA (Prinicle
Component Analysis to reduce more than 7100 genes to 23 features,
while minding to keep at least 90% of the variance.

The training was done with logistic regression, after reviewing several 
models and their accuracy levels.


Getting Started
------------

From within the repo directory run

`./CancerGene/back.py`

You can now see the accuracy level of the model.

-----
About Training & Dataset
--

The dataset was derived from Kaggle. It consists of 7100 gene expressions in the DNA, 
in 72 cancer patients.

Project Organization
------------

    ├── README.md                    <- The top-level README for developers using this project
    ├── LICENSE.md                   <- MIT
    ├── .gitignore                   <- For environment directories
    │
    ├── CancerGene                   <- Containing the software itself
    │   ├── back.py                  <- backend code
    │
    └── tests                        <- Tests directory, .gitignored
        └── backend_tests.py         <- Unit tests of backend
 
Dependencies
------------

- Python
- scikit-learn
- Pandas
- NumPy
- IMDB
--------
# CancerGene
