#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

import time # for timer
from utils import get_dataset, powersettool, shapley
from options import args_parser
from update import test_inference
from models import MLP


if __name__ == '__main__':
    start_time = time.time() # start the timer
    args = args_parser()
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'
    #print("device is", device) # print check

    powerset = list(powersettool(range(1,args.num_users+1))) #generate a powerset list of tuples eg [(),(1,),(2),(1,2)]
    # print(powerset) # print check
    # dictionary containing all the accuracies for the subsets
    accuracy_dict = {}
    accuracy_dict[()] = 0


    for subset in powerset[1:]:  #omit the first element which is ()
        # print(subset) # print check
        args.trueshapley = ''.join([str(i) for i in subset])
        # print('args.trueshapley is:', args.trueshapley) # print check
    
        # load datasets
        train_dataset, test_dataset, _ = get_dataset(args) # no user_groups required because this is the baseline
        #print("End print") # print check 

        # BUILD MODEL
        # if args.model == 'cnn':
        #     # Convolutional neural netork
        #     if args.dataset == 'mnist':
        #         global_model = CNNMnist(args=args)
        #     elif args.dataset == 'fmnist':
        #         global_model = CNNFashion_Mnist(args=args)
        #     elif args.dataset == 'cifar':
        #         global_model = CNNCifar(args=args)
        if args.model == 'mlp':
            # Multi-layer preceptron
            img_size = train_dataset[0][0].shape  # train_dataset is just a datasets object and not a dictionary of datasets
            len_in = 1
            for x in img_size:
                len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                                dim_out=args.num_classes)
        # else:
        #     exit('Error: unrecognized model')

        # Set the model to train and send it to device.
        global_model.to(device)
        global_model.train()

        print(global_model) 


        ### Training
        # Set optimizer and criterion
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                        momentum=0.5)
        # elif args.optimizer == 'adam':
        #     optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
        #                                 weight_decay=1e-4)

        trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        criterion = torch.nn.NLLLoss().to(device)
        # criterion = torch.nn.CrossEntropyLoss().to(device)
        epoch_loss = []

        for epoch in tqdm(range(args.epochs)):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(trainloader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = global_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
                    # print("shape of output:", outputs[0]) # print check
                    # print("shape of labels:", labels.shape) # print check
                    # print("loss is", loss) # print check
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch+1, batch_idx * len(images), len(trainloader.dataset),
                        100. * batch_idx / len(trainloader), loss.item()))
                batch_loss.append(loss.item())

            loss_avg = sum(batch_loss)/len(batch_loss)
            print('\nTrain loss:', loss_avg)
            epoch_loss.append(loss_avg)

        # # Plot loss
        # plt.figure()
        # plt.plot(range(len(epoch_loss)), epoch_loss)
        # plt.xlabel('epochs')
        # plt.ylabel('Train loss')
        # plt.savefig('../save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
        #                                              args.epochs))

        # testing
        test_acc, test_loss = test_inference(args, global_model, test_dataset)

        print('Test on', len(test_dataset), 'samples')
        print("Test Accuracy: {:.2f}%".format(100*test_acc))
        
        accuracy_dict[subset] = test_acc
    
    shapley_dict = shapley(accuracy_dict, args.num_users)
    
    totalRunTime = time.time()-start_time
    print('\n Total Run Time: {0:0.4f}'.format(totalRunTime))  # print total time

    # accuracy_file = open('../save/nn_{}_{}_{}_{}.txt'.format(args.dataset, args.model,
    #                         args.epochs, args.traindivision), 'a')
    # lines = ['Trainset: '+args.traindivision+'_'+args.trueshapley, '\n',
    #         'Accuracy: ' +str(test_acc), '\n', 
    #         'Total Run Time: {0:0.4f}'.format(totalRunTime),
    #         '\n']
    # accuracy_file.writelines(lines)
    # accuracy_file.close()


    accuracy_file = open('../save/nnshapley_{}_{}_{}_{}.txt'.format(args.dataset, args.model,
                            args.epochs, args.traindivision), 'a')
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
    
    