
import torch


def test(models, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()

    total = 0
    correct1 = 0
    correct2 = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores1, scores2 = models['backbone'](inputs)
            _, preds1 = torch.max(scores1.data, 1)
            _, preds2 = torch.max(scores2.data, 1)
            total += labels.size(0)
            correct1 += (preds1 == labels).sum().item()
            correct2 += (preds2 == labels).sum().item()
            acc1 = 100 * correct1 / total
            acc2 = 100 * correct2 / total
            acc = (100 * correct1 / total + 100 * correct2 / total) * 0.5

    return acc1, acc2, acc
