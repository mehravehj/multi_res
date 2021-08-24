# multi_res


**Alpha**

Code for multi resolution network 

Based on training alpha mixing weights using gradient descent (see DARTS)

Blocks: convolution (conv-relu-BN)

Double number of outout channels when downsampling

Dataset: CIFAR10

To run:

`train_multire.py -dd dataset_directory -tn test_name` 

**Tree**

Using a tree to sample a path through the network The path distribution is softmax(all_resolutions_weight) Resolution weight is updated using -loss with some miximg weight
