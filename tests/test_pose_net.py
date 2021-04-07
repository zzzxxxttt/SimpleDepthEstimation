import torch
import torch.nn as nn

from detectron2.config.config import CfgNode as CN
from detectron2.modeling.pose_net.GooglePoseNet import GooglePoseNet, GoogleMotionNet

if __name__ == '__main__':

    cfg = CN()
    cfg.MODEL = CN()
    cfg.MODEL.POSE_NET = CN()
    cfg.MODEL.POSE_NET.GROUP_NORM = False
    cfg.MODEL.POSE_NET.LEARN_SCALE = True
    cfg.MODEL.POSE_NET.MASK_MOTION = True
    cfg.MODEL.POSE_NET.USE_DEPTH = False


    def hook(self, input, output):
        print(output.data.cpu().numpy().shape)


    net = GoogleMotionNet(cfg).cuda()

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.register_forward_hook(hook)

    image = torch.randn(2, 3, 512, 512).cuda()
    context = torch.randn(2, 3, 512, 512).cuda()

    y = net(image, context)
    print([y_.shape for y_ in y])

    # out = net(torch.randn(1, 3, 512, 512), 1.0)
