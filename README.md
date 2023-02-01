# STRec: Sparse Transformer for Sequential Recommendations

This is the implementation of the paper "STRec: Sparse Transformer for Sequential Recommendations".

You can implement our model according to the following steps:

1. Make sure that you have installed recent versions of python, CUDA and pytorch.

2. Install the RecBole package using 'pip install recbole'.

3. Run the python file 'STRec.py' to implement our framework on ML-20M. A well-trained model is provided for ML-20M, you can  test the speed of it with the default 'STRec.yaml'. For the other datasets, you can change the config file to 'Go.yaml' for Gowalla and 'ame.yaml' for Amazon-Electronics. Note that the downloading of dataset in the first running may raise error, and you need to run the python file again after downloading.

4. Change the 'mode' option in the config file for different stages.
    'pre_train' for pre-training stage.
    'train' for fine-tuning stage.
    'speed' for efficiency test.
    'test' for evaluate on the test set.
    After the training stages, you need to set the direction of the '.pth' file of the models in the config file. And you need to set 'uni20' to replace 'full' in 'eval_args' for evaluating negative sampling.
   
5. Change the 'hash' option to 'True' in the config file for replacing the FFN with a hash function. The hash function should be defined manually, and we define one for ML-20M.

6. We also provide a baseline backbone model for comparison, use the 'Backbone.yaml' config file for running

