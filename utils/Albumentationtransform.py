import albumentations as A
import albumentations.pytorch as AP
import numpy as np

class AlbumentationTransforms:
    def __init__(self, transforms_list=[]):
        transforms_list.append(AP.ToTensorV2())
        self.transforms = A.Compose(transforms_list)

    def __call__(self, image):
        image = np.array(image)
        return self.transforms(image=image)['image']