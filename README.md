# 2D semantic segmentation multilabels with Unet
[![Build Status](https://jenkins.indigo-datacloud.eu/buildStatus/icon?job=Pipeline-as-code/DEEP-OC-org/UC-adnaneds-unet/main)](https://jenkins.indigo-datacloud.eu/job/Pipeline-as-code/job/DEEP-OC-org/job/UC-adnaneds-unet/job/main)

Semantic segmentation with Unet Deep Learning model applied to segment Cercospora Leaf Spot. The dataset used to train this model contains three classes: Background, Leaf and Disease.

To launch it, first install the package then run [deepaas](https://github.com/indigo-dc/DEEPaaS):
```bash
git clone https://github.com/adnaneds/unet
cd unet
pip install -e .
cd ..
deepaas-run --listen-ip 0.0.0.0
```
The associated Docker container for this module can be found in https://github.com/adnaneds/DEEP-OC-unet.

## Project structure
```
├── LICENSE                <- License file
│
├── README.md              <- The top-level README for developers using this project.
│
├── requirements.txt       <- The requirements file for reproducing the analysis environment, e.g.
│                             generated with `pip freeze > requirements.txt`
│
├── setup.py, setup.cfg    <- makes project pip installable (pip install -e .) so
│                             unet can be imported
│
├── unet    <- Source code for use in this project.
│   │
│   ├── __init__.py        <- Makes unet a Python module
│   │
│   └── api.py             <- Main script for the integration with DEEP API
│
└── Jenkinsfile            <- Describes basic Jenkins CI/CD pipeline
```
