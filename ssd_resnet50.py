import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os
import resnet50 as resnet

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes

        self.cfg = voc     #(coco, voc)[num_classes == 4]
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
        self.size = size
        # SSD network
        self.resnet = base
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect()

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        # apply resnet50-->ssd
        x = self.resnet.conv1(x)  ## 75*75
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)      
        x = self.resnet.maxpool(x) # 75*75
        # print(x.shape)
        x = self.resnet.layer1(x) #75*75
        x = self.resnet.layer2(x) #38*38
        sources.append(x)

        x = self.resnet.layer3(x) #19*19
        sources.append(x)
        x = self.resnet.layer4(x) #10*10
        sources.append(x)

        #extra layers
        for k, v in enumerate(self.extras): 
            x = v(x)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        # print(self.loc)
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        
        if self.phase == "test":

            output = self.detect.apply(self.num_classes, 0, 200, 0.01, 0.45,
                loc.view(loc.size(0), -1, 4),                  # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )

        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location='cuda:0'))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_bn_nopd(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv1_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


def add_extras(i):
    # Extra layers added to ResNet for feature scaling
    layers = []
    #conv14
    layers += [conv1_bn(i,256,1)]
    layers += [conv_bn(256,512,2)]
    #conv15
    layers += [conv1_bn(512,128,1)]
    layers += [conv_bn(128,256,2)]
    #conv16
    layers += [conv1_bn(256,128,1)]
    layers += [conv_bn_nopd(128,128,2)]
    return layers

def multibox(resnet50, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []

    ### may be have bug ###
    extras_source = [1, 3, 5]

    loc_layers += [nn.Conv2d(512, 4 * 4, kernel_size=1)]
    conf_layers += [nn.Conv2d(512, 4 * num_classes, kernel_size=1)]

    loc_layers += [nn.Conv2d(1024, 4 * 4, kernel_size=1)]
    conf_layers += [nn.Conv2d(1024, 4 * num_classes, kernel_size=1)]

    loc_layers += [nn.Conv2d(2048, 6 * 4, kernel_size=1)]
    conf_layers += [nn.Conv2d(2048, 6 * num_classes, kernel_size=1)]

    # for k, v in enumerate(extra_layers[1::2], 2):
    for k, v in enumerate(extras_source):
        k += 2
        loc_layers += [nn.Conv2d(extra_layers[v][0].out_channels,
                                 cfg[k] * 4, kernel_size=1)]
        conf_layers += [nn.Conv2d(extra_layers[v][0].out_channels,
                                  cfg[k] * num_classes, kernel_size=1)]
    return resnet50, extra_layers, (loc_layers, conf_layers)

extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256],
    '512': [],
}

mbox = {
    '300':[4, 6, 6, 6, 4, 4],
    '512': [],
}

def build_ssd(phase, size=300, num_classes=21):

    # add, no use
    size = 300

    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return

    base_, extras_, head_ = multibox(resnet.resnet50(pretrained=False), add_extras(2048),mbox[str(size)], num_classes)

    return SSD(phase, size, base_, extras_, head_, num_classes)


if __name__ =="__main__":
    torch.backends.cudnn.enabled = False
    ssd = build_ssd("train")
    x = torch.zeros((32, 96, 19, 19))
    x = ssd.loc[0](x)
    print(x.size())
    x = torch.zeros((32, 1280, 10, 10))
    x = ssd.loc[1](x)
    print(x.size())
    x = torch.zeros((32, 512, 5, 5))
    x = ssd.loc[2](x)
    print(x.size())
    x = torch.zeros((32, 256, 3, 3))
    x = ssd.loc[3](x)
    print(x.size())
    x = torch.zeros((32, 256, 2, 2))
    x = ssd.loc[4](x)
    print(x.size())
    x = torch.zeros((32, 128, 1, 1))
    x = ssd.loc[5](x)
    print(x.size())