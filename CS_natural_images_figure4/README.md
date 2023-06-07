Use this code to train any model from the results on compressive sensing for natural images in Figure 4 of the paper __Analyzing the Sample Complexity of Self-Supervised Image Reconstruction Methods__.

## Usage

To start training a model for a desired training set size and supervised or self-supervised training, provide the correct configartion file from the `experiment_configs` folder. For example
```
python run_CS_natural_images.py \
--config_file "figure4_sup_mu1_N100.txt" \
--path_to_ImageNet_train "../../../media/ssd1/ImageNet/ILSVRC/Data/CLS-LOC/" \
--experiment_number '001' \
--run_which_seeds 'run_best_seed'
```
trains the model with the best performing seed, supervised training and 100 training examples.

`experiment_number` just acts as an unique identifier for the folder holding the experimental results and for running all considered seeds for this training set size instead of only the best one replace `run_best_seed` with `run_all_seeds`.

To obtain the histogram of the variance of the stochastic gradients in Figure 4, first run
```
python train_network_for_histogram.py --path_to_ImageNet_train "../../../media/ssd1/ImageNet/ILSVRC/Data/CLS-LOC/"
```
then use `load_histogram_figure4.ipynb` to plot the histogram.
