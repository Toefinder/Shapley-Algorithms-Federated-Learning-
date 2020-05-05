# Shapley-Algorithms-Federated-Learning-
Efficient approximation algorithms for Shapley Values in Horizontal Enterprise Federated Learning

## This is the implementation of paper titled "Time Efficient Algorithms to approximate the Shapley Values for Horizontal Enterprise Federated Learning" 
Experiments are conducted on modified MNIST, called OurMNIST. The purpose is to measure how accurate and fast the approximation algorithms are. 


## Structure of the folder:
/data: contains the data for OurMNIST. These data are in the form of dictionaries. The keys correspond to the indices of users, and each value is a list of tuples (feature, label) where feature and label are tensors. To see how these dictionaries are generated, run Datasets.ipynb and Datasets_combinations.ipynb. 

/save: the folder that our generated output file will be saved to

/src: the folder containing all the code
	/src/models.py: the code for 2-layer MLP that we use. 
	/src/options.py: the specifications of all the arguments (eg number of local epochs, number of global epochs,...) we can run with 
	/src/utils.py: containing a definition of wrapper for OurMNIST dataset, and other useful functions 
	/src/update.py: the local update


## Requirements
Install the packages from requirements.txt

### Running the experiments
First, cd to the folder /src

To run Exact Non-federated Shapley for 5 users, for case 1 (equal distribution with same size), and 10 global epochs:
```python exactNonfederated_main.py --model=mlp --dataset=OurMNIST --traindivision=1 --epochs=10```
*Note that --traindivision is the case for dataset split

To run Exact Federated Shapley for 5 users, for case 1 (equal distribution with same size), and 10 global epochs:
```python exactFederated_main.py --model=mlp --dataset=OurMNIST --traindivision=1 --epochs=10```


To run OR, Adjusted OR, OR-TMC, Adjusted OR-TMC, just change the .py argument accordingly. Note that the MR algorithm is timed differently (the total run time is recorded, instead of just time for calculation of Shapley Value)

For 10 clients, add another argument --num_users=10, for example 
```python federatedOR_main.py --model=mlp --dataset=OurMNIST --num_users=10 --traindivision=1 --epochs=10```


## Options
The default values for various paramters parsed to the experiment are given in options.py. Details are given some of those parameters:
--dataset: Default: 'mnist'. Options: 'OurMNIST'
--model: Default: 'mlp'. 
--gpu: Default: None (runs on CPU). Can also be set to the specific gpu id.
--epochs: Number of rounds of training.
--lr: Learning rate set to 0.01 by default.
Federated Parameters
--num_users:Number of users. Default is 5. Options: 10
--local_ep: Number of local training epochs in each user. Default is 10.
--local_bs: Batch size of local updates in each user. Default is 64.
Other parameters: 
--traindivision: the case of dataset split (values are from 1 to 5) 
