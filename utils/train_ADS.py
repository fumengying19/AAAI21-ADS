
import torch.nn as nn
from utils.util import *
import torch.optim.lr_scheduler as lr_scheduler


class MyEntLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        p = x / torch.repeat_interleave(x.sum(dim=1).unsqueeze(-1), repeats=10, dim=1)
        logp = torch.log2(p)
        ent = -torch.mul(p, logp)
        entloss = torch.sum(ent, dim=1)
        return entloss


# step1
def train_init(models, criterion, optimizers, dataloaders):
    models['backbone'].train()
    for data in dataloaders['train']:
        inputs = data[0].cuda()
        labels = data[1].cuda()
        labels = get_one_hot_label(labels, nCls=10)

        optimizers['backbone'].zero_grad()

        for name, value in models['backbone'].named_parameters():
            value.requires_grad = True

        scores1, scores2 = models['backbone'](inputs)
        target_loss1 = criterion(scores1, labels)
        target_loss2 = criterion(scores2, labels)

        m_backbone_loss1 = torch.sum(target_loss1) / target_loss1.size(0)
        m_backbone_loss2 = torch.sum(target_loss2) / target_loss2.size(0)
        loss = m_backbone_loss1 + m_backbone_loss2

        loss.backward()
        optimizers['backbone'].step()

# step 2
def train_conv(cfg, models, cos_criterion, unlabeled_loader):
    models['backbone'].train()

    f1_name_list = ['conv5.weight', 'protos1']
    f2_name_list = ['conv6.weight', 'protos2']

    for i, (input_un, _) in enumerate(unlabeled_loader):
        input_un = input_un.cuda()

        output1_un, output2_un = models['backbone'](input_un)

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

        ent1_un = MyEntLoss().forward(output1_un)
        ent2_un = MyEntLoss().forward(output2_un)

        ent_un_cat = torch.cat((torch.unsqueeze(ent1_un, 1), torch.unsqueeze(ent2_un, 1)), dim=1)
        best_ent_un = torch.mean(ent_un_cat, dim=1)
        ent_un_transf = nn.Sigmoid()(best_ent_un - cfg.TRAIN.MIN_CLBR)
        min_disc_wt = (1 - ent_un_transf)

        min_disc_wt_avg = (min_disc_wt/128).detach()
        min_discrepantloss_un = torch.mean(torch.mul(min_disc_wt_avg, min_discrepantloss_un))

        loss = min_discrepantloss_un
        optimizer_conv.zero_grad()
        loss.backward()
        optimizer_conv.step()


# step 3
def train_f(cfg, models, criterion, criterion_re, dataloaders, unlabeled_loader):

    models['backbone'].train()

    f1_name_list = ['conv5.weight', 'protos1']
    f2_name_list = ['conv6.weight', 'protos2']

    input_list = list(enumerate(dataloaders['train']))
    len_input = len(input_list)
    for i, (input_un, _) in enumerate(unlabeled_loader):
        j = i % len_input
        (_, (inputs, target)) = input_list[j]
        inputs = inputs.cuda()
        input_un = input_un.cuda()
        target = target.cuda()

        target = get_one_hot_label(target, nCls=10)

        output1, output2 = models['backbone'](inputs)
        output1_un, output2_un = models['backbone'](input_un)

        # define  optimizer only for F1,F2
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

        ent1_un = MyEntLoss().forward(output1_un)
        ent2_un = MyEntLoss().forward(output2_un)

        ent_un_cat = torch.cat((torch.unsqueeze(ent1_un, 1), torch.unsqueeze(ent2_un, 1)), dim=1)
        best_ent_un = torch.mean(ent_un_cat, dim=1)
        ent_un_transf = nn.Sigmoid()(best_ent_un - cfg.TRAIN.MAX_CLBR)
        max_disc_wt = ent_un_transf

        max_disc_wt_avg = (max_disc_wt/128).detach()
        max_discrepantloss_un = torch.mean(torch.mul(max_disc_wt_avg, max_discrepantloss_un))

        loss = m_backbone_loss1 + m_backbone_loss2 + max_discrepantloss_un

        optimizer_f.zero_grad()
        loss.backward()
        optimizer_f.step()


def train(
        cfg, logger, models, criterion, cos_criterion, criterion_re,
        optimizers, schedulers, dataloaders, unlabeled_loader,
        num_epochs, cycle):
    logger.info('>> Train a Model.')

    for epoch in range(num_epochs):
        schedulers['backbone'].step()

        # train for one epoch
        if epoch == 0:
            # step1: Conv,F1,F2 -- label data
            train_init(models, criterion, optimizers, dataloaders)

        # step2: Conv -- unlabel
        train_conv(cfg, models, cos_criterion, unlabeled_loader)

        # step3: F1,F2 -- unlabel
        train_f(cfg, models, criterion, criterion_re, dataloaders, unlabeled_loader)

        # step1: Conv,F1,F2 -- label data
        train_init(models, criterion, optimizers, dataloaders)

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
