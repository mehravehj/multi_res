# multi_res


# alpha
Code for multi resolution network 

Based on training alpha mixing weights using gradient descent (see DARTS)

Blocks: convolution (conv-relu-BN)

Double number of outout channels when downsampling

Dataset: CIFAR10

To run:

`train_multire.py -dd dataset_directory -tn test_name` 
