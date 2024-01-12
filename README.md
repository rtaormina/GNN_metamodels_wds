## Assessing the performances and transferability of graph neural network metamodels for water distribution systems

### Overview
This repository contains the code implementation associated with the research paper titled "Assessing the performances and transferability of graph neural network metamodels for water distribution systems" By Kerimov, Bulat, et al.

- **DOI:** [https://doi.org/10.2166/hydro.2023.031]

## Getting Started

### Prerequisites

Make sure you have the following dependencies installed:

- `matplotlib==3.5.1`
- `networkx==2.6.3`
- `numpy==1.21.2`
- `PyYAML==6.0`
- `scikit_learn==1.0.2`
- `torch==1.10.1`
- `torch_geometric==2.0.3`
- `wntr==0.4.0`


You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Data
The dataset used in this research is available via Dropbox. You can download it using the following link:

```https://drive.google.com/drive/folders/1wtWFnv60K214CaXVBdXKQhL3GyxWVnhN?usp=sharing```

After downloading, place the dataset files in the data/ folder within this repository.

### Running experiments

To run the experiments, you can use the following jupyter notebooks:

- `main_GNN_vs_MLP.ipynb`: This notebook is used to run the experiments for the GNN vs MLP comparison.


### Citation

```bash
@article{10.2166/hydro.2023.031,
    author = {Kerimov, Bulat and Bentivoglio, Roberto and Garzón, Alexander and Isufi, Elvin and Tscheikner-Gratl, Franz and Steffelbauer, David Bernhard and Taormina, Riccardo},
    title = "{Assessing the performances and transferability of graph neural network metamodels for water distribution systems}",
    journal = {Journal of Hydroinformatics},
    volume = {25},
    number = {6},
    pages = {2223-2234},
    year = {2023},
    month = {10},
    issn = {1464-7141},
    doi = {10.2166/hydro.2023.031},
    url = {https://doi.org/10.2166/hydro.2023.031},
    eprint = {https://iwaponline.com/jh/article-pdf/25/6/2223/1334433/jh0252223.pdf},
}
```


