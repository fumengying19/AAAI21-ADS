"""Active Learning Procedure in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
"""

import random
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from data.sampler import SubsetSequentialSampler
# Custom
import models.resnet as resnet
from utils.config import get_configs
from utils.dataset import get_dataset, get_transform, get_training_functions_single
from utils.train_ADS import *
from utils.test_ADS import *
from utils.util import *
from torch.backends import cudnn
import os
import torch
import numpy as np
import time
from torch.utils.data import DataLoader
# from torch.nn import functional


class Myl1loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        # return torch.nn.functional.l1_loss(x1, x2)
        return torch.mean(abs(x1 - x2), dim=1).cuda()


class Myrel1loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        # return torch.tensor(1).cuda() - torch.nn.functional.l1_loss(x1, x2)
        return torch.tensor(1).cuda() - torch.mean(abs(x1 - x2), dim=1).cuda()


def get_uncertainty(umodels, uunlabeled_loader):
    umodels['backbone'].eval()
    uuncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for (inputs, _) in uunlabeled_loader:
            inputs = inputs.cuda()

            output1, output2 = umodels['backbone'](inputs)

            ent1 = MyEntLoss().forward(output1)
            ent2 = MyEntLoss().forward(output2)
            ent_cat = torch.cat((torch.unsqueeze(ent1, 1), torch.unsqueeze(ent2, 1)), dim=1)
            pred_loss = torch.mean(ent_cat, dim=1)

            uuncertainty = torch.cat((uuncertainty, pred_loss), 0)

    return uuncertainty.cpu()


# Main
if __name__ == '__main__':

    # hyper params
    port = 9999
    dataset = 'cifar10'
    method = 'random'
    num_classes = 10
    cfg = get_configs(port=port)

    # path and logger
    time_str = time.strftime('%m%d%H%M%S', time.localtime())
    dataset_root = cfg.DATASET.ROOT[dataset]
    checkpoint_dir = os.path.join('output', time_str)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    log_save_file = checkpoint_dir + "/log.txt"
    my_logger = Logger(method, log_save_file).get_log
    my_logger.info("Checkpoint Path: {}".format(checkpoint_dir))
    print_config(cfg, my_logger)

    # define dataset and dataloaders
    train_transform, test_transform = get_transform(dataset)
    train_dataset, test_dataset, unlabeled_dataset = get_dataset(
        dataset_root, dataset, train_transform, test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=cfg.TRAIN.BATCH)

    Performance = np.zeros((3, 10))
    for trial in range(cfg.ACTIVE_LEARNING.TRIALS):
        # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000 data points
        # from the entire dataset.
        indices = list(range(cfg.DATASET.NUM_TRAIN))
        random.shuffle(indices)
        labeled_set = indices[:cfg.ACTIVE_LEARNING.ADDENDUM]
        unlabeled_set = indices[cfg.ACTIVE_LEARNING.ADDENDUM:]

        random.shuffle(unlabeled_set)

        train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH,
                                  sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=True)
        dataloaders = {'train': train_loader, 'test': test_loader}

        # Model
        resnet18 = resnet.ResNet18(num_classes=num_classes).cuda()
        models = {'backbone': resnet18}
        torch.backends.cudnn.benchmark = True
        num_images = 50000
        initial_budget = 1000
        all_indices = set(np.arange(num_images))
        initial_indices = random.sample(all_indices, initial_budget)
        sampler = torch.utils.data.sampler.SubsetRandomSampler(initial_indices)

        # Active learning cycles
        for cycle in range(cfg.ACTIVE_LEARNING.CYCLES):
            # Randomly sample 10000 unlabeled data points
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:cfg.ACTIVE_LEARNING.SUBSET]

            # Create unlabeled dataloader for the unlabeled subset
            unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=cfg.TRAIN.BATCH,
                                          sampler=SubsetSequentialSampler(subset),
                                          pin_memory=True)

            criterion, optimizers, schedulers = get_training_functions_single(
                cfg, models
            )
            cos_criterion = Myl1loss()
            criterion_re = Myrel1loss()

            # Training and test
            minw_ent_clbr = train(
                cfg, my_logger, models, criterion, cos_criterion, criterion_re, optimizers,
                schedulers, dataloaders, unlabeled_loader,
                cfg.TRAIN.EPOCH, cycle
                # querry_dataloader, task_model, optim_task_model, val_loader
            )
            torch.save(
                        {
                            'cycle': cycle,
                            'state_dict_backbone': models['backbone'].state_dict()
                        },
                        '{}/active_resnet18_cifar10_cycle_9epoch_200.pth'.format(checkpoint_dir)
                    )
            my_logger.info("Start Test")
            acc1,  acc2, acc = test(models, dataloaders, mode='test')
            my_logger.info("End Test")
            # print(acc1, acc2, acc)
            my_logger.info('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(
                trial + 1, cfg.ACTIVE_LEARNING.TRIALS, cycle + 1,
                cfg.ACTIVE_LEARNING.CYCLES, len(labeled_set), acc)
            )
            Performance[trial, cycle] = acc

            # update dataloaders
            # Measure uncertainty of each data points in the subset
            # uncertainty = get_uncertainty(models, unlabeled_loader)

            # Index in ascending order
            # arg = np.argsort(uncertainty)

            # # Update the labeled dataset and the unlabeled dataset, respectively
            # budget = cfg.ACTIVE_LEARNING.ADDENDUM
            # labeled_set += list(torch.tensor(subset)[arg][-budget:].numpy())
            # unlabeled_set =\
            #     list(torch.tensor(subset)[arg][:-budget].numpy()) + unlabeled_set[cfg.ACTIVE_LEARNING.SUBSET:]
            #
            # # Create a new dataloader for the updated labeled dataset
            # dataloaders['train'] = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH,
            #                                   sampler=SubsetRandomSampler(labeled_set),
            #                                   pin_memory=True)
            # sampler = torch.utils.data.sampler.SubsetRandomSampler(labeled_set)

            # save a ckpt
            torch.save(
                        {
                            'cycle': cycle,
                            'state_dict_backbone': models['backbone'].state_dict()
                        },
                        '{}/active_resnet18_cifar10_cycle{}_epoch_199.pth'.format(checkpoint_dir, cycle)
                    )
        # np.save(checkpoint_dir + '/l_set.npy', np.array(labeled_set))

    my_logger.info("Performance summary: ")
    my_logger.info("Trail 1: {}".format(Performance[0]))
    my_logger.info("Trail 2: {}".format(Performance[1]))
    my_logger.info("Trail 3: {}".format(Performance[2]))
