## Random Noise Warm-up for Uncertainty Calibration

Jeonghwan Cheon*, Se-Bum Paik† 

\* First author: jeonghwan518@kaist.ac.kr  
† Corresponding author: sbpaik@kaist.ac.kr  


### Associated article

This repository contains the implementation and demo codes for the manuscript "**Random Noise Warm-up for Uncertainty Calibration**" (currently under review). The preprint is available on arXiv.

[![arXiv](https://img.shields.io/badge/arXiv-2412.17411-b31b1b.svg)](https://arxiv.org/abs/2412.17411)

#### Abstract

Uncertainty calibration, the process of aligning confidence with accuracy, is a hallmark of human intelligence. However, most machine learning models struggle to achieve this alignment, particularly when the training dataset is small relative to the network's capacity. Here, we demonstrate that uncertainty calibration can be effectively achieved through a pretraining method inspired by developmental neuroscience. Specifically, training with random noise before data training allows neural networks to calibrate their uncertainty, ensuring that confidence levels are aligned with actual accuracy. We show that randomly initialized, untrained networks tend to exhibit erroneously high confidence, but pretraining with random noise effectively calibrates these networks, bringing their confidence down to chance levels across input spaces. As a result, networks pretrained with random noise exhibit optimal calibration, with confidence closely aligned with accuracy throughout subsequent data training. These pre-calibrated networks also perform better at identifying "unknown data" by exhibiting lower confidence for out-of-distribution samples. Our findings provide a fundamental solution for uncertainty calibration in both in-distribution and out-of-distribution contexts.

### Simulations and experiments

The detailed and core implementation of the model can be found in the ```/src``` directory. We provide demonstration code and a corresponding trained model for experiments using CIFAR-10 on ResNet-18 in the ```/demo``` directory.

#### Random noise warm-up and training models
```
python demo/train_model.py
```
The code includes three tasks: (1) random noise warm-up training, (2) task learning with warm-up trained networks, and (3) task learning from conventional random initialization without random warm-up. The final model weights are saved for calibration and out-of-distribution detection analysis. For a short demonstration, you can use the provided trained model weights without running this script.

#### Evaluation of model calibration
```
python demo/evaluate_calibration.py
```

The code measures model calibration and includes post-calibration methods such as temperature scaling, vector scaling, and isotonic regression.

#### Evaluation of out-of-distribution detection
```
python demo/evaluate_ood_detection.py
```

The code evaluates out-of-distribution detection performance. It also includes out-of-distribution detection with various detection frameworks and post-processing methods, such as temperature scaling, ODIN, and energy score.

### Citation

```bibtex
@article{cheon2024pretraining
  title={Pretraining with random noise for uncertainty calibration},
  author={Cheon, Jeonghwan and Paik, Se-Bum},
  journal={arXiv preprint arXiv:2412.17411},
  year={2024}
}
```
