import os
import argparse
import yaml
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from dataset import NIHDataset
from net import chest_net
from nih_loss import NIHLoss
from utils import TrainLogger, Evaluator

from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix',
                        type=str,
                        help='prefices for logs, checkpoints')
    parser.add_argument('--gpu', type=str, help='specify which GPU to use')

    parser.add_argument('--cfg',
                        type=str,
                        default='./config/defualt.cfg',
                        help='config file. see readme')
    parser.add_argument('--weights_path',
                        type=str,
                        default=None,
                        help='weights file')
    parser.add_argument('--n_cpu',
                        type=int,
                        default=16,
                        help='number of workers')

    parser.add_argument('--checkpoint',
                        type=str,
                        help='pytorch checkpoint file path')
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default='checkpoints',
                        help='directory where checkpoint files are saved')

    parser.add_argument('--tfboard',
                        help='tensorboard path for logging',
                        type=str,
                        default='logs')

    return parser.parse_args()


def main():
    args = parse_args()
    print('Setting Arguments...', args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(os.path.join(args.checkpoint_dir, args.prefix), exist_ok=True)

    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f)

    print(cfg)

    batch_size = cfg['TRAIN']['BATCH_SIZE']
    base_lr = cfg['MODEL']['LR'] / batch_size
    schedule = eval(cfg['TRAIN']['SCHEDULE'])
    weight_decay = eval(cfg['TRAIN']['DECAY'])
    max_epoch = cfg['TRAIN']['MAX_EPOCH']
    valid_interval = cfg['INTERVAL']

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010))
    ])
    valid_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = NIHDataset(
        cfg['PATH']['TRAIN']['ENTRY_FILE'],
        transforms=train_transforms,
        mode='train',
    )

    subtrain_dataset = NIHDataset(
        cfg['PATH']['VALID']['ENTRY_FILE_A'],
        transforms=valid_transforms,
        mode='valid',
    )

    valid_dataset = NIHDataset(
        cfg['PATH']['VALID']['ENTRY_FILE_B'],
        transforms=valid_transforms,
        mode='valid',
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=24,
    )

    subtrain_loader = DataLoader(
        subtrain_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=24,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=24,
    )

    n_images = train_dataset.n_images

    def burnin_schedule(i, burn_in=1000):
        if i < burn_in:
            # factor = pow(i / burn_in, 3)
            factor = 1.
        elif i < schedule[0] * n_images / batch_size:
            factor = 1.
        elif i < schedule[1] * n_images / batch_size:
            factor = 0.1
        elif i < schedule[2] * n_images / batch_size:
            factor = 0.01
        return factor

    net = chest_net(cfg)
    if args.weights_path:
        pass
    elif args.checkpoint:
        print("loading pytorch ckpt...", args.checkpoint)
        state = torch.load(args.checkpoint)
        if 'model_state_dict' in state.keys():
            net.load_state_dict(state['model_state_dict'])
        else:
            net.load_state_dict(state)
    net.cuda()

    criterion = NIHLoss(batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr, amsgrad=True)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

    params_dict = dict(net.named_parameters())
    params = []
    for key, value in params_dict.items():
        if 'conv.weight' in key:
            params += [{
                'params': value,
                'weight_decay': weight_decay / batch_size
            }]
        else:
            params += [{'params': value, 'weight_decay': 0.0}]

    logger = SummaryWriter(os.path.join(args.tfboard, args.prefix))
    iter_train_loader = iter(train_loader)

    max_iter = n_images * max_epoch // batch_size
    epoch = 1
    train_logger = TrainLogger(
        max_iter,
        max_epoch,
        train_loader,
        criterion,
        logger,
    )
    
    for i_step in range(1, max_iter + 1, 1):
        try:
            _, images, target = next(iter_train_loader)

        except StopIteration:
            # start eval
            epoch += 1
            train_logger = TrainLogger(
                max_iter,
                max_epoch,
                train_loader,
                criterion,
                logger,
            )

            if epoch % valid_interval == 1:

                min_loss = np.inf
                train_evaluator = Evaluator(
                    i_step,
                    net,
                    subtrain_loader,
                    criterion,
                    logger,
                    args.prefix,
                    mode='train',
                )
                valid_evaluator = Evaluator(
                    i_step,
                    net,
                    valid_loader,
                    criterion,
                    logger,
                    args.prefix,
                    mode='valid',
                )

                if valid_evaluator.loss < min_loss:
                    print('seva model...')
                    print(f'f1 {valid_evaluator.f1}')
                    print(f'acc {valid_evaluator.acc}')
                    print(f'loss {valid_evaluator.loss}')
                    min_loss = valid_evaluator.loss
                    torch.save(
                        {
                            'iter': i_step,
                            'epoch': epoch,
                            'f1': valid_evaluator.f1,
                            'loss': valid_evaluator.loss,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                        },
                        os.path.join(args.checkpoint_dir, args.prefix,
                                     f'best_model.ckpt'))

            iter_train_loader = iter(train_loader)
            _, images, target = next(iter_train_loader)

        images = Variable(images).cuda().float()
        target = Variable(target).cuda().float()

        net.train()
        optimizer.zero_grad()
        pred = net(images)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        current_lr = scheduler.get_lr()[0] * batch_size
        train_logger.update(i_step, pred, target, current_lr, epoch)

    logger.close()


if __name__ == '__main__':
    main()
