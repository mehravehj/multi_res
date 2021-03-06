# multi_res
## Requirements
`
Python >= 3.5.5, PyTorch == 0.3.1, torchvision == 0.2.0
` 

## Alpha

Code for multi resolution network 

Based on training alpha (mixing weights) using gradient descent (see DARTS)

Blocks: convolution (conv-relu-BN)

Double the number of outout channels when downsampling
Zero padding along channel dimension to match

Dataset: CIFAR10

To run:

`
train_multire.py -dd dataset_directory -tn test_name
` 
## Tree

Using a tree to sample a path through the network The path distribution is softmax(all_resolutions_weight). Resolution weight is updated using -loss with some mixing weight

Blocks: convolution (conv-relu-BN)

Dataset: CIFAR10

To run:

`
train_multire_tree.py -dd dataset_directory -tn test_name
` 
