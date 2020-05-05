#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP
from utils import get_dataset, average_weights, exp_details, powersettool, shapley, \
                  calculate_gradients, update_weights_from_gradients


if __name__ == '__main__':


    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    
    # BUILD MODEL
    # if args.model == 'cnn':
    #     # Convolutional neural network
    #     if args.dataset == 'mnist':
    #         global_model = CNNMnist(args=args)
    #     elif args.dataset == 'fmnist':
    #         global_model = CNNFashion_Mnist(args=args)
    #     elif args.dataset == 'cifar':
    #         global_model = CNNCifar(args=args)

    if args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[1][0][0].shape  
        
        # print(img_size) # print check 

        len_in = 1
        for x in img_size:
            len_in *= x
        # print('number of dimension_in is :', len_in) # print check
        global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)    
    # else:
    #     exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()    ### create 2^5 submodels based on this 

    ######## Timing starts ########
    start_time = time.time() # start the timer

    # Powerset list 
    powerset = list(powersettool(range(1,args.num_users+1))) # generate a powerset list of tuples eg [(),(1,),(2),(1,2)]

    # Initialize all the sub-models
    submodel_dict = {}  
    submodel_dict[()] = copy.deepcopy(global_model)

    for subset in range(1, args.num_users+1): # only form the submodels based on individual datasets
        submodel_dict[(subset,)] = copy.deepcopy(global_model)
        submodel_dict[(subset,)].to(device)
        submodel_dict[(subset,)].train()
    

    # # # test dictionary -- to ensure that the adjusted algo is correct
    # test_submodel_dict = {}
    # for subset in powerset[:-1]:    
    #     test_submodel_dict[subset] = copy.deepcopy(global_model)
    #     test_submodel_dict[subset].to(device)
    #     test_submodel_dict[subset].train()

    # dictionary containing all the accuracies for the subsets    
    accuracy_dict = {}
    # accuracy_dict[()] = 0

    totalRunTime = time.time() - start_time
    ######## Timing ends ########

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    ### number of users participating in the training round
    # m = max(int(args.frac * args.num_users), 1) 
    m = args.num_users #default is 5

    ### randomly choose that many users to participate
    # idxs_users = np.random.choice(range(args.num_users), m, replace=False) 
    idxs_users = np.arange(1, m+1)

    # print('user indexes are:', idxs_users,'type', type(idxs_users)) # print check

    # List of fraction of data from each user
    total_data = sum(len(user_groups[i]) for i in range(1,m+1))
    fraction = [len(user_groups[i])/total_data for i in range(1,m+1)]
    print("data fraction is:", fraction) # print check 

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()

        # note that the keys for train_dataset are [1,2,3,4,5]
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset[idx],
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights, fraction) 
        # print("global_weights is", global_weights) # print check




        # # test if the functions are behaving correctly
        # average_gradients = average_weights(gradients, fraction) # average_weights can be used to average gradients
        # global_weights2 = update_weights_from_gradients(average_gradients, global_model.state_dict()) 
        # print("global_weights2 is", global_weights) # print check


        loss_avg = sum(local_losses) / len(local_losses)

        train_loss.append(loss_avg)

        ######## Timing starts ########
        start_time = time.time() # start the timer   
        
        gradients = calculate_gradients(local_weights, global_model.state_dict()) 

        # update sub-model weights
        for i in idxs_users: 
            # update only the submodels (x,) for x being a participant
            subset_weights = update_weights_from_gradients(gradients[i-1], submodel_dict[(i,)].state_dict()) 
            submodel_dict[(i,)].load_state_dict(subset_weights)

        totalRunTime += time.time() - start_time
        ######## Timing ends ########

        # update global weights
        global_model.load_state_dict(global_weights) ### update the 2^n submodels as well 

        # Calculate avg training accuracy over all users at every epoch. 
        # For this case, since all users are participating in training, we need to adjust the code
        # list_acc, list_loss = [], []
        global_model.eval()
        # for c in range(args.num_users): (this doesn't apply in our case)
        # for idx in idxs_users:
        #     local_model = LocalUpdate(args=args, dataset=train_dataset[idx],
        #                               idxs=user_groups[idx], logger=logger)
        #     acc, loss = local_model.inference(model=global_model)
        #     list_acc.append(acc)
        #     list_loss.append(loss)
        # train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            # print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    # print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))




    ######## Timing starts ########
    start_time = time.time() # start the timer 

    # accuracy for the full dataset is the same as global accuracy
    accuracy_dict[powerset[-1]] = test_acc

    # the new algo
    for subset in powerset[:-1]: 
        if len(subset) > 1:
            # calculate the average of the subset of weights from list of all the weights
            subset_weights = average_weights([submodel_dict[(i,)].state_dict() for i in subset], [fraction[i-1] for i in subset]) 
            submodel = copy.deepcopy(submodel_dict[()])
            submodel.load_state_dict(subset_weights)
            
            test_acc, test_loss = test_inference(args, submodel, test_dataset)
            # print(f' \n Results after {args.epochs} global rounds of training (for OR): ') # print check
            # print("|---- Test Accuracy for {}: {:.2f}%".format(subset, 100*test_acc)) # print check
            accuracy_dict[subset] = test_acc
        else: 
            test_acc, test_loss = test_inference(args, submodel_dict[subset], test_dataset)
            accuracy_dict[subset] = test_acc
    # shapleystart_time = time.time() # print check
    
    # Test inference for the sub-models in submodel_dict

    # shapleytime = time.time() - shapleystart_time # print check
    # print(shapleytime) # print check 
 
    
    shapley_dict = shapley(accuracy_dict, args.num_users)

    
    
    totalRunTime += time.time() - start_time
    ######## Timing ends ########           


    # Saving the objects train_loss and train_accuracy:
    #file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    #    format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #           args.local_ep, args.local_bs)

    #with open(file_name, 'wb') as f:
    #    pickle.dump([train_loss, train_accuracy], f)

    
    print('\n Total Time for Shapley: {0:0.4f}'.format(totalRunTime))  # print total time

    #write information into a file 
    accuracy_file = open('../save/ORfedAdjusted_{}_{}_{}_{}_{}users.txt'.format(args.dataset, args.model,
                            args.epochs, args.traindivision, args.num_users), 'a')
    for subset in powerset:
        accuracy_lines = ['Trainset: '+args.traindivision+'_'+''.join([str(i) for i in subset]), '\n',
                'Accuracy: ' +str(accuracy_dict[subset]), '\n',
                '\n']
        accuracy_file.writelines(accuracy_lines)
    for key in shapley_dict:
        shapley_lines = ['Data contributor: '+str(key),'\n',
                'Shapley value: '+ str(shapley_dict[key]), '\n',
                '\n']
        accuracy_file.writelines(shapley_lines) 

    lines = ['Total Run Time: {0:0.4f}'.format(totalRunTime),
            '\n']
    accuracy_file.writelines(lines)
    accuracy_file.close()

    # PLOTTING (optional)
    #import matplotlib
    #import matplotlib.pyplot as plt
    #matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    
    # Plot Average Accuracy vs Communication rounds
    #plt.figure()
    #plt.title('Average Accuracy vs Communication rounds')
    #plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    #plt.ylabel('Average Accuracy')
    #plt.xlabel('Communication Rounds')
    #plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #            format(args.dataset, args.model, args.epochs, args.frac,
    #                   args.iid, args.local_ep, args.local_bs))
