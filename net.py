import torch
import torch.nn as nn
import torchvision.models as models


def chest_net(cfg):
    net = models.densenet121(pretrained=True)
    classifier = nn.Sequential(
        nn.Linear(1024, cfg['MODEL']['N_CLASS']),
        nn.Sigmoid()
    )
    net.classifier = classifier
    net = torch.nn.DataParallel(net)
    return net
