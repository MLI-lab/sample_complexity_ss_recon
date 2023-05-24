Use this code to train any model that is part of the results on image denoising in Figure 2 of the paper __Analyzing the Sample Complexity of Self-Supervised Image Reconstruction Methods__.

## Usage

Figure 2 contains our results for the performance of models trained with noise2noise [1], noisier2noise [2] and neighbor2neighbor [3] type loss functions as a function of the training set size.

To start training a model for one of the methods and a desired training set size choose the correct configuration file from the options folder and load it in the respective `config_run.ipynb` notebook.

## References
[1] Lehtinen et al. "Noise2Noise: Learning Image Restoration without Clean Data". In: Proceedings of the 35th International Conference on Machine Learning, PMLR, 2018.

[2] Moran et al. "Noisier2Noise: Learning to Denoise From Unpaired Noisy Data". In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020.

[3] Huang et al. "Neighbor2Neighbor: Self-Supervised Denoising From Single Noisy Images". In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021.
