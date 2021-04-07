import torch
import torch.nn as nn

from detectron2.config.config import CfgNode as CN
from detectron2.modeling.depth_net.GoogleResNet import GoogleResNet

if __name__ == '__main__':

    cfg = CN()
    cfg.MODEL = CN()
    cfg.MODEL.DEPTH_NET = CN()
    cfg.MODEL.DEPTH_NET.ENCODER_NAME = '18'
    cfg.MODEL.DEPTH_NET.UPSAMPLE_DEPTH = True
    cfg.MODEL.DEPTH_NET.FLIP_PROB = 0.5
    cfg.MODEL.DEPTH_NET.LEARN_SCALE = True


    def hook(self, input, output):
        print(output.data.cpu().numpy().shape)


    net = GoogleResNet(cfg).cuda()

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.register_forward_hook(hook)

    data = {'image': torch.randn(2, 3, 512, 512).cuda()}
    y = net(data)
    print(y['depth_pred'][0].shape)

    # out = net(torch.randn(1, 3, 512, 512), 1.0)