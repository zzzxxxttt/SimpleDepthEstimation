import torch
import torchvision.transforms as transforms

from .build import PREPROCESS_REGISTRY, Preprocess


@PREPROCESS_REGISTRY.register()
class ToTensor(Preprocess):
    def __init__(self, cfg):
        super(ToTensor, self).__init__(cfg)
        self.to_tensor = transforms.ToTensor()

    def forward(self, data_dict):
        for key in data_dict:
            if key in ['img', 'img_orig']:
                data_dict[key] = self.to_tensor(data_dict[key])
            elif key in ['ctx_img', 'ctx_img_orig']:
                data_dict[key] = [self.to_tensor(img) for img in data_dict[key]]
        return data_dict
