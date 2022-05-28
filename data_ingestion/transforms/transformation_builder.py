from typing import List, Dict, Any

from data_ingestion.transforms import NAME_TRANSFORMATION_MAP


def build_transform(transform_list: List[Dict[str, Any]]):
    """Build one callable object that serialy applies each transformation
    defined in ´transform_list´.
    :param transform_list: List of transformation to serially apply.
    """

    def get_funct(transform):
        """Retrieves a function from its configuration name. It also pops its
        name from the configuration"""
        return NAME_TRANSFORMATION_MAP[transform.pop("name")]

    transforms = [get_funct(transform) for transform in transform_list]
    return Compose(transforms, transform_list)


class Compose:
    def __init__(self, transforms, transform_params):
        self.transforms = transforms
        self.transform_params = transform_params

    def __call__(self, img):

        for t, p in zip(self.transforms, self.transform_params):

            img = t(img, **p)

        return img
