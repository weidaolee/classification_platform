import torch
import torch.nn as nn
import torch.nn.functional as F


class NIHLoss(nn.Module):
    def __init__(self, batch_size):
        super(NIHLoss, self).__init__()
        self.batch_size = batch_size
        self.stats = torch.tensor([
            0.10309489832322512,
            0.024759186585800928,
            0.0416250445950767,
            0.020540492329646807,
            0.11877452729218695,
            0.022440242597217268,
            0.015037459864430967,
            0.0020246164823403494,
            0.17743489118801284,
            0.05156974669996432,
            0.05646628612201213,
            0.030190866928291118,
            0.012763110952550838,
            0.04728861933642526,
        ]).cuda()
        self.tags = [
            'Atelectasis',
            'Cardiomegaly',
            'Consolidation',
            'Edema',
            'Effusion',
            'Emphysema',
            'Fibrosis',
            'Hernia',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pleural_Thickening',
            'Pneumonia',
            'Pneumothorax',
        ]
        self.loss_dict = {k: 0. for k in self.tags + ['Total']}

    def forward(self, pred, target):

        stats = self.stats.expand_as(target)
        weight = torch.abs(target - stats)

        total_loss = 0
        for i in range(len(self.tags)):
            self.loss_dict[self.tags[i]] = F.binary_cross_entropy(
                pred[:, i],
                target[:, i],
                weight[:, i],
            ) / len(self.tags)
            total_loss += self.loss_dict[self.tags[i]]
        self.loss_dict['Total'] = total_loss
        return self.loss_dict['Total']
