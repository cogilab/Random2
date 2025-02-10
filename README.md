## Pretraining with Random Nose for Uncertainty Calibration

arXiv Preprint arXiv:2412.17411  
Jeonghwan Cheon*, Se-Bum Paik† 

\* First author: jeonghwan518@kaist.ac.kr  
† Corresponding author: sbpaik@kaist.ac.kr  


### Associated article

This repository contains the implementation and demo codes for the manuscript "**Pretraining with Random Nose for Uncertainty Calibration**" (currently under submission). The preprint is available on arXiv.

[![arXiv](https://img.shields.io/badge/arXiv-2412.17411-b31b1b.svg)](https://arxiv.org/abs/2412.17411)

#### Abstract

Uncertainty calibration, the process of aligning confidence with accuracy, is a hallmark of human intelligence. However, most machine learning models struggle to achieve this alignment, particularly when the training dataset is small relative to the network's capacity. Here, we demonstrate that uncertainty calibration can be effectively achieved through a pretraining method inspired by developmental neuroscience. Specifically, training with random noise before data training allows neural networks to calibrate their uncertainty, ensuring that confidence levels are aligned with actual accuracy. We show that randomly initialized, untrained networks tend to exhibit erroneously high confidence, but pretraining with random noise effectively calibrates these networks, bringing their confidence down to chance levels across input spaces. As a result, networks pretrained with random noise exhibit optimal calibration, with confidence closely aligned with accuracy throughout subsequent data training. These pre-calibrated networks also perform better at identifying "unknown data" by exhibiting lower confidence for out-of-distribution samples. Our findings provide a fundamental solution for uncertainty calibration in both in-distribution and out-of-distribution contexts.

#### Research highlights

- Deep neural networks often struggle to properly calibrate both accuracy and confidence.
- Pretraining with random noise aligns confidence levels with actual accuracy.
- Random noise pretraining facilitates meta-learning by aligning confidence with the chance level.
- Pre-calibrated networks assess both known and unknown datasets through confidence estimation.

### Simulations and experiments

Each simulation and experimental result can be replicated by running the corresponding demo files in the ```\demo``` directory. The detailed and core implementation of the model can be found in the ```\src``` directory.

#### Prerequisites

- Python 3.11 (Python software foundation)
- Pytorch 2.1 + CUDA 11.8
- NumPy 1.26.0
- SciPy 1.11.4


### Citation

```bibtex
@article{cheon2024pretraining
  title={Pretraining with random noise for uncertainty calibration},
  author={Cheon, Jeonghwan and Paik, Se-Bum},
  journal={arXiv preprint arXiv:2412.17411},
  year={2024}
}
```
