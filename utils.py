import os
import torch
import pandas as pd


class TrainLogger():
    def __init__(self, max_iter, max_epoch, data_loader, criterion, logger):

        self.losses = {k: 0. for k in criterion.loss_dict.keys()}

        self.max_iter = max_iter
        self.max_epoch = max_epoch
        self.data_loader = data_loader
        self.criterion = criterion
        self.logger = logger

    def update(self, i_step, pred, target, lr, epoch):
        pred = (pred >= 0.5).float()
        target = target.data.float()

        # let c for class name
        for c in list(self.criterion.loss_dict.keys()):
            self.losses[c] = self.criterion.loss_dict[c]

        total_loss = self.losses['Total']
        if i_step % 5 == 0:
            print(
                f'''[Stept {i_step} / {self.max_iter}] [Epoch {epoch} / {self.max_epoch}] [Lr {lr}]  [Loss: total {total_loss}]'''
            )

            _losses = {}
            for k in list(self.criterion.loss_dict.keys()):
                _losses[k] = self.losses[k]

            self.logger.add_scalars(f'running/loss', _losses, i_step)


class Evaluator():
    def __init__(self, i_step, net, data_loader, criterion, logger, prefix,
                 mode):
        max_iter = len(data_loader)
        res_dir = os.path.join('results', prefix)
        os.makedirs(res_dir, exist_ok=True)

        net.eval()  # Set model to evaluation mode
        losses = {k: 0. for k in criterion.loss_dict.keys()}

        metrics = {
            m: {
                k: 1e-12
            }
            for m in ['acc', 'precision', 'recall', 'f1']
            for k in criterion.loss_dict.keys()
        }
        tp = {k: 1e-12 for k in criterion.loss_dict.keys()}
        tn = {k: 1e-12 for k in criterion.loss_dict.keys()}
        fp = {k: 1e-12 for k in criterion.loss_dict.keys()}
        fn = {k: 1e-12 for k in criterion.loss_dict.keys()}

        # Iterate over data.
        with torch.set_grad_enabled(False):
            for i, (name, images, target) in enumerate(data_loader):
                batch_size = target.size(0)
                n_class = target.size(1)

                images = images.cuda()
                target = target.cuda().float()

                pred = net(images)
                loss = criterion(pred, target)

                pred = (pred >= 0.5).float()
                target = target.data.float()

                # let j for index of class
                for j in range(len(criterion.tags)):
                    c = criterion.tags[j]
                    p = pred[:, j]
                    t = target[:, j]

                    _tp = (p * t).sum()
                    _tn = ((1 - p) * (1 - t)).sum()
                    _fp = (p * (1 - t)).sum()
                    _fn = ((1 - p) * t).sum()

                    tp[c] += _tp
                    tn[c] += _tn
                    fp[c] += _fp
                    fn[c] += _fn
                    losses[c] += criterion.loss_dict[c]

                    tp['Total'] += _tp
                    tn['Total'] += _tn
                    fp['Total'] += _fp
                    fn['Total'] += _fn
                losses['Total'] += criterion.loss_dict['Total']

                # save predicted results
                res = pd.DataFrame(columns=[
                    'Path',
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
                ])
                res['Path'] = name
                res[[
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
                ]] = pred.cpu().numpy()

                res.to_csv(os.path.join(res_dir, f'{i_step}.csv'),
                           index=False,
                           mode='a')

            for c in list(criterion.loss_dict.keys()):
                metrics['acc'][c] = (tp[c] + tn[c]) / (tp[c] + tn[c] + fp[c] +
                                                       fn[c])
                metrics['precision'][c] = (tp[c] / (tp[c] + fp[c]))
                metrics['recall'][c] = (tp[c] / (tp[c] + fn[c]))
                metrics['f1'][
                    c] = 2 * metrics['precision'][c] * metrics['recall'][c] / (
                        metrics['precision'][c] + metrics['recall'][c] + 1)

            acc = metrics['acc']['Total']
            precision = metrics['precision']['Total']
            recall = metrics['recall']['Total']
            f1 = metrics['f1']['Total']
            total_loss = losses['Total'] / max_iter

        for c in list(criterion.loss_dict.keys()):
            losses[c] /= max_iter

        print(
            f'''End Eval [Loss: total {total_loss}] [Metrics: acc {acc} precision {precision} recall {recall} f1 {f1}]'''
        )

        logger.add_scalars(f'{mode}/loss', losses, i_step)
        for m in ['acc', 'precision', 'recall', 'f1']:
            logger.add_scalars(f'{mode}/{m}', metrics[m], i_step)

        self.acc = acc
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.loss = total_loss
