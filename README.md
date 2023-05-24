# Analyzing_Sample_Complexity_Self_Supervised_Image_Reconstruction

Each folder contains the code to reproduce the results in one of the Figures 1,2,4,5 in the main boday of the paper __Analyzing the Sample Complexity of Self-Supervised Image Reconstruction Methods__.

In particular
- Figure 1: Simulations for subspace denoising
- Figure 2: Image denoising
- Figure 4: Compressive sensing for natural images
- Figure 5: Compressive sensing accelerated MRI

## Requirements
CUDA-enabled GPU is necessary to run the code. We tested this code using:
- Ubuntu 20.04
- CUDA 11.5
- Python 3.7.11
- PyTorch 1.10.0

## Installation
First, install PyTorch 1.10.0 with CUDA support following the instructions [here](https://pytorch.org/get-started/previous-versions/).
Then, to install the necessary packages run
```bash
pip install -r requirements.txt
```
We used the bart toolbox to pre-compute the sensitivity maps for the experiments on accelerated MRI. Install bart toolbox by following the instructions on their [home page](https://mrirecon.github.io/bart/).

## Datasets
### fastMRI
FastMRI is an open dataset, however you need to apply for access at https://fastmri.med.nyu.edu/. To run the experiments from our paper, you need to download the fastMRI brain dataset.

### ImageNet
ImageNet is an open dataset, and you can request access at https://image-net.org/download.php. To run the experiments from our paper, you need to download the ImageNet train set.

## Acknowledgments and references
The code for MRI reconstruction partly builds on the [fastMRI repository]( https://github.com/facebookresearch/fastMRI), and the code for image denoising on [Robust And Interpretable Blind Image Denoising Via Bias-Free Convolutional Neural Networks](https://github.com/LabForComputationalVision/bias_free_denoising).

- Zbontar et al. "fastMRI: An Open Dataset and Benchmarks for Accelerated MRI". In: https://arxiv.org/abs/1811.08839* (2018).
- Russakovsky et al. "ImageNet Large Scale Visual Recognition Challenge". In: *International Journal of Computer Vision* (2015).

# Analyzing_Sample_Complexity_of_Self_Supervised_Image_Reconstruction
# Analyzing_Sample_Complexity_of_Self_Supervised_Image_Reconstruction
