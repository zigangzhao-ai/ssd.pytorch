'''
code by zzg@2021/05/23
'''
# import data.coco.COCO_ROOT as COCO_ROOT
# import data.coco.COCODetection as COCODetection
from data import *
from utils.augmentations import SSDAugmentation, SSDAugmentation_mosaic
from layers.modules import MultiBoxLoss
import os
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import argparse
import random
import warnings
warnings.filterwarnings('ignore')
from attention.attention import WarmUpLR
import math
from ssd_resnet50 import build_ssd

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MEANS = (104, 117, 123)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

model_save_dir = "weights/resnet"

if not os.path.exists(model_save_dir):
      os.makedirs(model_save_dir)

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--resnet_pre_model', default="weights/resnet50.pth", #"mb2-imagenet-71_8.pth",
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')

parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],MEANS))
       
    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        ssd_net.resnet.load_state_dict(torch.load(args.resnet_pre_model), strict=False)

        if isinstance(ssd_net.resnet, torch.nn.DataParallel):
            ssd_net.mobilenet = ssd_net.resnet.module

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    warmup_scheduler = WarmUpLR(optimizer, 40 * 5)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True, worker_init_fn=worker_init_fn)
    # create batch iterator
    batch_iterator = iter(data_loader)

    t0 = time.time()
    for iteration in range(args.start_iter, cfg['max_iter']):

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
        
        if iteration != 0 and (iteration % epoch_size == 0):
             epoch += 1
        # load train data    
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        with torch.no_grad():
            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda()) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann) for ann in targets]

        # backprop
        optimizer.zero_grad()
        # forward
        out = net(images)
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()

        if iteration <= 200:
            warmup_scheduler.step()
        else:
            optimizer.step()

        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        if iteration % 2 == 0:
            #print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')
            print('[%3d/%d] iter '%(epoch+args.start_iter/epoch_size,int(cfg['max_iter']) / epoch_size) + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')
            print('timer: %.4f sec.' % (time.time() - t0))
            t0 = time.time()
        #if iteration != 0 and iteration % 5000 == 0:
        if iteration >= 5000 and iteration % 500 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), model_save_dir + '/'+ 'resent50_' +
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(), model_save_dir + '/' + 'resent50_final' + '.pth')

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def xavier(param):
    init.xavier_uniform_(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        # m.bias.data.zero_()
        # torch.nn.init.constant_(m.bias.data, 0.0)
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def setup_seed(seed=2019):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed) #gpu
    np.random.seed(seed) #numpy
    random.seed(seed)
    torch.backends.cudnn.deterministic=True # cudnn

def worker_init_fn(worker_id): # After creating the workers, each worker has an independent seed that is initialized to the curent random seed + the id of the worker
    np.random.seed(np.random.get_state()[1][0] + worker_id)

if __name__ == '__main__':
    # torch.backends.cudnn.enabled = True
    setup_seed() # set ramdom seed
    train()
