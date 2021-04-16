
import torch.nn as nn
# from utils.test_net import test
from utils.util import *
import torch.optim.lr_scheduler as lr_scheduler
# import matplotlib.pyplot as plt
# from ptflops import get_model_complexity_info


class MyEntLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # p = torch.nn.functional.softmax(x, dim=1)
        p = x / torch.repeat_interleave(x.sum(dim=1).unsqueeze(-1), repeats=10, dim=1)
        logp = torch.log2(p)
        ent = -torch.mul(p, logp)
        entloss = torch.sum(ent, dim=1)
        # entloss = torch.mean(entloss) + torch.tensor(0.01).cuda()
        return entloss


# step1
def train_init(models, criterion, optimizers, dataloaders):
    models['backbone'].train()
    # global iters

    for data in dataloaders['train']:
        inputs = data[0].cuda()
        labels = data[1].cuda()
        labels = get_one_hot_label(labels, nCls=10)
        # cfg.global_iter.update_vars()

        # iters += 1

        optimizers['backbone'].zero_grad()

        for name, value in models['backbone'].named_parameters():
            value.requires_grad = True

        scores1, scores2 = models['backbone'](inputs)
        # flops, params = get_model_complexity_info(models['backbone']
        # , (3, 32, 32), as_strings=True, print_per_layer_stat=True)
        target_loss1 = criterion(scores1, labels)
        target_loss2 = criterion(scores2, labels)

        m_backbone_loss1 = torch.sum(target_loss1) / target_loss1.size(0)
        m_backbone_loss2 = torch.sum(target_loss2) / target_loss2.size(0)
        loss = m_backbone_loss1 + m_backbone_loss2

        loss.backward()
        optimizers['backbone'].step()

        # Visualize
        # if cfg.global_iter.iter % 100 == 0:
        #     logger.info("Trial {}, Cycle {}, Epoch {}, Loss_module {}".format(
        #         trial + 1, cycle + 1, epoch + 1,
        #         round(loss.item(), 4))
        #     )

    # print("train-init-finish")


# step 2
def train_conv(cfg, models, cos_criterion, unlabeled_loader
               # , minw_ent_clbr
               ):
    models['backbone'].train()
    # global iters

    f1_name_list = ['conv5.weight', 'protos1']
    f2_name_list = ['conv6.weight', 'protos2']

    input_unlist = list(enumerate(unlabeled_loader))

    # plot entropy distribution
    # a = torch.load('./output/cifar10/train/active_resnet18_cifar10_trial0.pth')
    # models['backbone'].load_state_dict(a['state_dict_backbone'])
    # ent_list = torch.tensor([]).cuda()

    for i, (_, _) in enumerate(unlabeled_loader):
        (_, (input_un, _)) = input_unlist[i]
        # inputs = inputs.cuda()
        input_un = input_un.cuda()
        # target = target.cuda()

        # iters += 1

        # output1, output2 = models['backbone'](inputs)
        output1_un, output2_un = models['backbone'](input_un)

        # step2: Conv -- min dis -- unlabel

        # define  optimizer only for conv
        for name, value in models['backbone'].named_parameters():
            if name in f1_name_list:
                value.requires_grad = False
            elif name in f2_name_list:
                value.requires_grad = False
            else:
                value.requires_grad = True
        params = filter(lambda p: p.requires_grad, models['backbone'].parameters())
        optimizer_conv = torch.optim.SGD(params, lr=cfg.TRAIN.LR, momentum=0.9, weight_decay=0.0005)
        sched_conv = lr_scheduler.MultiStepLR(optimizer_conv, milestones=cfg.TRAIN.MILESTONES)
        sched_conv.step()

        min_discrepantloss_un = cos_criterion(output1_un, output2_un)

        # baseline need change class: mycosloss
        ent1_un = MyEntLoss().forward(output1_un)
        ent2_un = MyEntLoss().forward(output2_un)

        ent_un_cat = torch.cat((torch.unsqueeze(ent1_un, 1), torch.unsqueeze(ent2_un, 1)), dim=1)
        best_ent_un = torch.mean(ent_un_cat, dim=1)
        # best_ent_un = torch.max(ent_un_cat, dim=1)[0]
        # best_ent_un = torch.min(ent_un_cat, dim=1)[0]
        # ent_list = torch.cat((ent_list, best_ent_un.detach()))
        ent_un_transf = nn.Sigmoid()(best_ent_un - cfg.TRAIN.MIN_CLBR)
        min_disc_wt = (1 - ent_un_transf)  # / 3 + torch.cosine_similarity(output1_un, output2_un) / 3 + torch.\
        # cosine_similarity(ent1_un.unsqueeze(1), ent2_un.unsqueeze(1)) / 3

        min_disc_wt_avg = (min_disc_wt/128).detach()
        min_discrepantloss_un = torch.mean(torch.mul(min_disc_wt_avg, min_discrepantloss_un))

        loss = min_discrepantloss_un
        optimizer_conv.zero_grad()
        loss.backward()
        optimizer_conv.step()

        # Visualize
        # if cfg.global_iter.iter % 100 == 0:
        #     logger.info("Trial {}, Cycle {}, Epoch {}, Loss_module {}".format(
        #         trial + 1, cycle + 1, epoch + 1,
        #         round(loss.item(), 4))
        #     )
    # print("train-conv-finish")
    # c = b.detach().numpy()
    # plt.hist(c, bins=50, facecolor="blue", edgecolor="black")
    # plt.savefig('1.jpg')
    # d = 1
    # minw_ent_clbr = ent_list.sort()[0][-round(0.1*ent_list.shape[0])]
    # return minw_ent_clbr


# step 3
def train_f(cfg, models, criterion, criterion_re, dataloaders, unlabeled_loader):

    models['backbone'].train()
    # global iters

    f1_name_list = ['conv5.weight', 'protos1']
    f2_name_list = ['conv6.weight', 'protos2']

    # input_unlist = list(enumerate(unlabeled_loader))
    input_list = list(enumerate(dataloaders['train']))
    len_input = len(input_list)
    # for i, (inputs, target) in enumerate(dataloaders['train']):
    for i, (input_un, _) in enumerate(unlabeled_loader):
        j = i % len_input
        # (_, (input_un, _)) = input_unlist[i]
        (_, (inputs, target)) = input_list[j]
        inputs = inputs.cuda()
        input_un = input_un.cuda()
        target = target.cuda()

        target = get_one_hot_label(target, nCls=10)
        # cfg.global_iter.update_vars()

        # iters += 1

        output1, output2 = models['backbone'](inputs)
        output1_un, output2_un = models['backbone'](input_un)

        # step3: F1,F2 -- max dis -- unlabel

        # define  optimizer only for conv
        for name, value in models['backbone'].named_parameters():
            if name in f1_name_list:
                value.requires_grad = True
            elif name in f2_name_list:
                value.requires_grad = True
            else:
                value.requires_grad = False
        params = filter(lambda p: p.requires_grad, models['backbone'].parameters())
        optimizer_f = torch.optim.SGD(params, lr=cfg.TRAIN.LR, momentum=0.9, weight_decay=0.0005)
        sched_f = lr_scheduler.MultiStepLR(optimizer_f, milestones=cfg.TRAIN.MILESTONES)
        sched_f.step()

        taskloss_1 = criterion(output1, target)
        taskloss_2 = criterion(output2, target)
        m_backbone_loss1 = torch.sum(taskloss_1) / taskloss_1.size(0)
        m_backbone_loss2 = torch.sum(taskloss_2) / taskloss_2.size(0)

        max_discrepantloss_un = criterion_re(output1_un, output2_un)
        # baseline need change class: mycosloss
        ent1_un = MyEntLoss().forward(output1_un)
        ent2_un = MyEntLoss().forward(output2_un)

        ent_un_cat = torch.cat((torch.unsqueeze(ent1_un, 1), torch.unsqueeze(ent2_un, 1)), dim=1)
        best_ent_un = torch.mean(ent_un_cat, dim=1)
        # ent_list = torch.cat((ent_list, best_ent_un.detach()))
        ent_un_transf = nn.Sigmoid()(best_ent_un - cfg.TRAIN.MAX_CLBR)
        max_disc_wt = ent_un_transf  # / 3 + torch.cosine_similarity(output1_un, output2_un) / 3 + torch.\
        # cosine_similarity(ent1_un.unsqueeze(1), ent2_un.unsqueeze(1)) / 3

        max_disc_wt_avg = (max_disc_wt/128).detach()
        max_discrepantloss_un = torch.mean(torch.mul(max_disc_wt_avg, max_discrepantloss_un))

        loss = m_backbone_loss1 + m_backbone_loss2 + max_discrepantloss_un

        optimizer_f.zero_grad()
        loss.backward()
        optimizer_f.step()

        # Visualize
        # if cfg.global_iter.iter % 100 == 0:
        #     logger.info("Trial {}, Cycle {}, Epoch {}, Loss_module {}".format(
        #         trial + 1, cycle + 1, epoch + 1,
        #         round(loss.item(), 4))
        #     )

    # print("train-F-finish")


def train(
        cfg, logger, models, criterion, cos_criterion, criterion_re,
        optimizers, schedulers, dataloaders, unlabeled_loader,
        num_epochs, cycle  # , minw_ent_clbr, cycle, checkpoint_dir
        # querry_dataloader, task_model, optim_task_model, val_loader
):
    logger.info('>> Train a Model.')
    # best_acc = 0.

    for epoch in range(num_epochs):
        schedulers['backbone'].step()

        # train_epoch(
        #     cfg,
        #     logger,
        #     models,
        #     criterion, cos_criterion, criterion_re,
        #     optimizers,
        #     dataloaders,
        #     epoch, tri, cyc,
        #     epoch_loss
        # )
        # train for one epoch
        if epoch == 0:
            # step1: Conv,F1,F2 -- label data
            train_init(models, criterion, optimizers, dataloaders)
        # print("initial already")

        # step2: Conv -- unlabel
        if epoch < 50 and cycle < 10:
            train_conv(cfg, models, cos_criterion, unlabeled_loader)
        # print("conv already")
        # print(minw_ent_clbr)

        # step3: F1,F2 -- unlabel
            train_f(cfg, models, criterion, criterion_re, dataloaders, unlabeled_loader)
        # print("f already")

        # step1: Conv,F1,F2 -- label data
        train_init(models, criterion, optimizers, dataloaders)
        # print("init already")

        # step4: task model -- only labeled data
        # train_task(querry_dataloader, task_model, optim_task_model)
        # print("task already")

        # task model evaluate on validation set
        # validate(val_loader, task_model, logger)
        # print("val already")

        # if epoch in [0, 49, 199] and cycle in [0, 1, 2, 9]:
        #     # Save a checkpoint
        #     torch.save(
        #         {
        #             'cycle': cycle,
        #             'epoch': epoch,
        #             'state_dict_backbone': models['backbone'].state_dict()
        #         },
        #         '{}/active_resnet18_cifar10_cycle{}_epoch{}.pth'.format(checkpoint_dir, cycle, epoch)
        #     )
        # Save a checkpoint
    #     if epoch > cfg.TRAIN.EPOCH-20:
    #         logger.info("Epoch: {}".format(epoch))
    #         acc1, acc2, acc = test(models, dataloaders, mode='test')
    #         # print(acc1, acc2, acc)
    #         if best_acc < acc:
    #             best_acc = acc
    # #             torch.save({
    # #                     'epoch': epoch + 1,
    # #                     'state_dict_backbone': models['backbone'].state_dict()
    # #                 }, '%s/active_resnet18_cifar10.pth' % checkpoint_dir
    # #             )
    #         logger.info('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
    logger.info('>> Finished.')


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
