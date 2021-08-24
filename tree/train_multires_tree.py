from __future__ import print_function

import argparse
import os
from datetime import datetime
from os import path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from data_loader_tree import data_loader
from model_tree import multires_model

from tree import Quaternary_Tree, Tree_Config, sample_with_probability, update_path_prob

parser = argparse.ArgumentParser(description='PyTorch Mutiresolution Training')
parser.add_argument('--dataset', '-d', default='CIFAR10', type=str, help='dataset name')
parser.add_argument('--channels', '-c', type=str, default='16', help='number of channels for 1st resolution')
parser.add_argument('--leng', '-l', type=int, default=6, help='depth of network')
parser.add_argument('--batchsize', '-b', type=int, default=200, help='batch size')
parser.add_argument('--test_name', '-tn', type=int, default=100, help='test name for saving model')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--epochs', '-e', type=int, default=100, help='epochs to train')
parser.add_argument('--validation_percent', '-vp', type=float, default=0.5, help='percent of train data for validation')

parser.add_argument('--learning_rate', '-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--min_learning_rate', '-mlr', default=0.001, type=float, help='min learning rate')
parser.add_argument('--weight_momentum', '-wm', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', '-wd', default=0, type=float, help='weight decay')

parser.add_argument('--max_scales', '-mx', type=int, default=4, help='number of scales to use')

parser.add_argument('--data_dir', '-dd', type=str, default='./data/', help='dataset directory')
parser.add_argument('--workers', '-wr', type=int, default=0, help='number of workers to load data')

parser.add_argument('-mixw', default=0.1, type=float, help='mixing weight for path probability update')

args = parser.parse_args()


def main():
    startTime = datetime.now()
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    print('Test ', args.test_name)
    print('-----------------------')
    print('test parameters')
    print(args)
    print('-----------------------')
    ncat = 10 # number of categories
    net = multires_model(ncat=ncat, channels=args.channels, leng=args.leng) # create model
    net.cuda()
    print('-----------------------')
    print('Model:')
    print(net)

    criterion = nn.CrossEntropyLoss() # classification loss criterion
    criterion = criterion.cuda()

    weight_parameters= parameters(net) # get network and alpha parameters
    weight_optimizer = optim.SGD(weight_parameters, lr=args.learning_rate, momentum=args.weight_momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(weight_optimizer, args.epochs+1, eta_min=args.min_learning_rate) # scheduler for weight learning rate

    conv_tree = Quaternary_Tree() #create tree
    conv_tree.init_Tree(Tree_Config(args.max_scales, args.leng))

    save_dir = './checkpoint/ckpt_' + str(args.test_name) + '.t7' #checkpoint save directory
    if path.exists(save_dir): # load from checkpoint if it exists
        best_model, current_epoch, loss_progress, accuracy_progress, best_epoch, best_accuracy, index, conv_tree\
            = load_checkpoint(save_dir, net, weight_optimizer, scheduler)
    else: # initialize checkpoint parameters if not resuming training
        best_model, current_epoch, loss_progress, accuracy_progress, best_epoch, best_accuracy, index\
            = initialize_save_parameters(net)

    # loading dataset
    if path.exists(args.data_dir):
        dataset_dir = args.data_dir
    else:
        dataset_dir = '~/Desktop/codes/multires/data/' # my default dataset directory

    train_loader, validation_loader, test_loader, indices, num_class = data_loader(args.dataset, args.validation_percent, args.batchsize,
                                                                                   indices=index,
                                                                                   dataset_dir=dataset_dir,
                                                                                   workers=args.workers)

    # training
    for epoch in range(current_epoch+1, args.epochs+1):
        print('epoch ', epoch)
        print('net learning rate: ', weight_optimizer.param_groups[0]['lr'])
        # train
        train_loss, validation_loss, train_accuracy, validation_accuracy = train_valid(conv_tree, train_loader, validation_loader, net, weight_optimizer, criterion, epoch)
        scheduler.step()
        if epoch % 5 == 0 or epoch == args.epochs: # test and save checkpoint every 5 epochs
            print('Testing path: ')
            print(net.path_id)
            test_loss, test_accuracy = test(test_loader, net)
            # record loss
            loss_progress['train'].append(train_loss)
            loss_progress['validation'].append(validation_loss)
            loss_progress['test'].append(test_loss)
            # record accuracy
            accuracy_progress['train'].append(train_accuracy)
            accuracy_progress['validation'].append(validation_accuracy)
            accuracy_progress['test'].append(test_accuracy)
            if test_accuracy > best_accuracy: # update best test accuracy
                print('--------------> Best accuracy')
                # record model with the best test accuracy
                best_model = {}
                for key in net.state_dict():
                    best_model[key] = net.state_dict()[key].clone().detach()
                best_accuracy = test_accuracy
                best_epoch = epoch

            print('train accuracy: ', train_accuracy, ' ....... validation accuracy: ', validation_accuracy, ' ....... test accuracy: ', test_accuracy)
            print('best accuracy:', best_accuracy,' at epoch ', best_epoch)
            print('...........SAVING............')
            save_checkpoint(save_dir, net, best_model, weight_optimizer, scheduler, epoch, loss_progress,
                            accuracy_progress, best_epoch, best_accuracy, index, conv_tree) # save checkpoint
        print('Training time: ', datetime.now() - startTime) # print  time from the start of training


def initialize_save_parameters(model):
    epoch = 0
    index = 0
    best_epoch = 0
    best_model = 0
    best_accuracy = 0
    loss_progress = {'train': [], 'validation': [], 'test': []}
    accuracy_progress = {'train': [], 'validation': [], 'test': []}
    return best_model, epoch, loss_progress, accuracy_progress, best_epoch, best_accuracy, index


def load_checkpoint(save_dir, model, weight_optimizer, scheduler):
    print('Loading from checkpoint...')
    checkpoint = torch.load(save_dir)
    epoch = checkpoint['epoch']
    loss_progress = checkpoint['loss_progress']
    accuracy_progress = checkpoint['accuracy_progress']
    best_model = checkpoint['best_model']
    best_epoch = checkpoint['best_epoch']
    best_accuracy = checkpoint['best_accuracy']
    index = checkpoint['indices']
    tree = checkpoint['tree']

    model.load_state_dict(checkpoint['model'])
    weight_optimizer.load_state_dict(checkpoint['weight_optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    return best_model, epoch, loss_progress, accuracy_progress, best_epoch, best_accuracy, index, tree


def save_checkpoint(save_dir, model, best_model, weight_optimizer, scheduler, epoch, loss_progress, accuracy_progress, best_epoch, best_accuracy, index, tree):
    state = {
        'test_properties': vars(args),
        'seed': args.seed,
        'indices': index,
        'best_epoch': best_epoch,
        'best_model': best_model,
        'loss_progress': loss_progress,
        'accuracy_progress': accuracy_progress,
        'best_accuracy': best_accuracy,
        'model': model.state_dict(),
        'epoch': epoch,
        'weight_optimizer': weight_optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'tree' : tree,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, save_dir)


def calculate_accuracy(logits, target, cul_total=0, cul_prediction=0):
    _, test_predicted = logits.max(1)
    test_total = target.size(0)
    correct_prediction = test_predicted.eq(target).sum().item()
    cul_prediction += correct_prediction
    cul_total += test_total
    return cul_prediction, cul_total


def parameters(model):
    all_parameters = [y for x, y in model.named_parameters()]
    sum_param = sum([np.prod(p.size()) for p in all_parameters])
    print('Number of model parameters: ', sum_param)

    return all_parameters


def train_valid(tree, train_queue, validation_queue, model, weight_optimizer, criterion=nn.CrossEntropyLoss(), epoch=0):
    model.train()
    train_loss = 0
    validation_loss = 0

    train_correct = 0
    validation_correct = 0

    train_total = 0
    validation_total = 0

    train_accuracy = 0
    validation_accuracy = 0
    validation_iterator = iter(validation_queue)
    for batch_idx, (train_inputs, train_targets) in enumerate(train_queue):
        #sample path
        path, path_id, sampling_function = sample_with_probability(tree) # sample path
        model.set_path(path_id, sampling_function) #set path in model
        #train
        train_inputs, train_targets = train_inputs.cuda(), train_targets.cuda()
        weight_optimizer.zero_grad()
        train_outputs = model(train_inputs)
        train_minibatch_loss = criterion(train_outputs, train_targets)
        train_minibatch_loss.backward()
        weight_optimizer.step()
        train_loss += train_minibatch_loss.detach().cpu().item()
        train_correct, train_total = calculate_accuracy(train_outputs, train_targets, train_total, train_correct)
        #validation
        validation_inputs, validation_targets = next(validation_iterator) # only works for vp 0.5, otherwise use next(iter(validation_iterator))
        validation_inputs, validation_targets = validation_inputs.cuda(), validation_targets.cuda()
        validation_outputs = model(validation_inputs)
        validation_minibatch_loss = criterion(validation_outputs, validation_targets)
        validation_loss += validation_minibatch_loss.detach().cpu().item()
        validation_correct, validation_total = calculate_accuracy(validation_outputs, validation_targets, validation_total, validation_correct)
        update_path_prob(path, -validation_minibatch_loss.detach().cpu().item(), mixing_weight=args.mixw) #update path probability with -loss
    validation_accuracy = validation_correct / validation_total
    train_loss = train_loss / (batch_idx + 1)
    validation_loss = validation_loss / (batch_idx + 1)
    train_accuracy = train_correct / train_total

    return train_loss, validation_loss, train_accuracy, validation_accuracy

def test(test_queue, model, criterion=nn.CrossEntropyLoss()):
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_idx, (test_inputs, test_targets) in enumerate(test_queue):
            test_inputs, test_targets = test_inputs.cuda(), test_targets.cuda()
            test_outputs = model(test_inputs)
            test_minibatch_loss = criterion(test_outputs, test_targets)
            test_loss += test_minibatch_loss.detach().cpu().item()
            test_correct, test_total = calculate_accuracy(test_outputs, test_targets, test_total, test_correct)
    test_loss = test_loss / (batch_idx + 1)
    return test_loss, test_correct/test_total

def sample_tree(tree):
    path, sampling_function = sample_with_probability(tree)
    return path, sampling_function


if __name__ == '__main__':
  main()
