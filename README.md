# FairGrad
Official implementation of *Fair Resource Allocation in Multi-Task Learning*.

![Toy Example](/misc/toy.png)

## Supervised Learning
The performance is evaluated under 3 scenarios:
 - Image-level Classification. The [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset contains 40 tasks.
 - Regression. The QM9 dataset contains 11 tasks, which can be downloaded automatically from Pytorch Geometric.
 - Dense Prediction. The [NYU-v2](https://github.com/lorenmt/mtan) dataset contains 3 tasks and the [Cityscapes](https://github.com/lorenmt/mtan) dataset contains 2 tasks.

### Setup Environment
Following [Nash-MTL](https://github.com/AvivNavon/nash-mtl) and [FAMO](https://github.com/Cranial-XIX/FAMO), we implement our method with the `MTL` library.

First, create the virtual environment:
```
conda create -n mtl python=3.9.7
conda activate mtl
python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113
```

Then, install the repo:
```
git clone https://github.com/OptMN-Lab/fairgrad.git
cd fairgrad
python -m pip install -e .
```

### Run Experiment
The dataset by default should be put under `experiments/EXP_NAME/dataset/` folder where `EXP_NAME` is chosen from `{celeba, cityscapes, nyuv2, quantum_chemistry}`. To run the experiment:
```
cd experiments/EXP_NAME
sh run.sh
```

## Reinforcement Learning
The experiments are conducted on [Meta-World](https://github.com/Farama-Foundation/Metaworld) benchmark. To run the experiments on `MT10` and `MT50` (the instructions below are partly borrowed from [CAGrad](https://github.com/Cranial-XIX/CAGrad)):

1. Create python3.6 virtual environment.
2. Install the [MTRL](https://github.com/facebookresearch/mtrl) codebase.
3. Install the [Meta-World](https://github.com/Farama-Foundation/Metaworld) environment with commit id `d9a75c451a15b0ba39d8b7a8b6d18d883b8655d8`.
4. Copy the `mtrl_files` folder to the `mtrl` folder in the installed mtrl repo, then 

```
cd PATH_TO_MTRL/mtrl_files/ && chmod +x mv.sh && ./mv.sh
```

5. Follow the `run.sh` to run the experiments.
