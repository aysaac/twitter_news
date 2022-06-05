import os
import logging
from typing import Optional, Union

import torch
import torch.nn as nn

from .model_definition.fusion import ToxtliModel
from .model_definition.resnet_50 import load_resnet
from .model_definition.resnet18 import MyResNet18
from .model_definition.CDCNs import CDCN,CDCN_classifier_1
from .model_definition.siamese import SiameseNet
from .model_definition.triplet import TripletNet, FeatureExtractor


Model = torch.nn.Module


def get_model(
    kind: str,
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    **kwargs: Union[int, bool],
) -> Model:
    """Retrieves a model.
    :param kind: One of {resnet_50, siamese, vision_resnet_18, triplet,
        fusion}.
    :param model_name: Defines the pretrained model to load.
    :param model_path: Path to some model weights to use
    """
    if kind == "BERT_model":
        return get_resnet_model(
            model_name=model_name, model_path=model_path, **kwargs
        )
    elif kind == "attention_net":
        return get_resnet_18(model_path=model_path)
    elif kind == "LTSM":
        return get_siamese_model(model_path=model_path, **kwargs)
    # elif kind == 'triplet':
    #     return get_triplet_model(model_name=model_name, **kwargs)
    elif kind == "bi-LTSM":
        return get_fusion(
            model_name=model_name, model_path=model_path, **kwargs
        )
    elif kind == "CDCN":
        return get_CDCN(model_path=model_path, **kwargs)
    elif kind == "CDCN_classifier_1":
        return get_CDCN_clasifier_1(model_path=model_path, **kwargs)
    else:
        raise ValueError(f"Unsupported model kind: {kind}")


def get_fusion(model_name=None, model_path=None, **kwargs):
    """Retrieves fusion model."""
    if model_name:
        model = ToxtliModel()
        file_dir = os.path.dirname(os.path.abspath(__file__))
        pre_trained_path = os.path.join(file_dir, "pre_trained")
        model_path = os.path.join(pre_trained_path, model_name)
        if os.path.isfile(model_path):
            # checkpoint = torch.load(model_path, map_location='cpu')
            # model.load_state_dict(checkpoint['model_state_dict'])
            model.load_state_dict(
                torch.load(model_path, map_location=torch.device("cpu"))
            )
            return model
        else:
            raise FileNotFoundError(f"No such model: {model_path}")
    elif model_path:
        raise NotImplementedError()
    else:
        model = ToxtliModel()
        return model


def get_resnet_model(model_name=None, model_path=None, **kwargs):
    """Retrieves resnet_50 model loaded in CPU"""
    if model_name:
        model = get_resnet_model()
        file_dir = os.path.dirname(os.path.abspath(__file__))
        pre_trained_path = os.path.join(file_dir, "pre_trained")
        model_path = os.path.join(pre_trained_path, model_name)
        if os.path.isfile(model_path):
            model.load_state_dict(
                torch.load(model_path, map_location=torch.device("cpu"))
            )
            return model
        else:
            raise FileNotFoundError(f"{model_name} not found")
    elif model_path:
        if os.path.isfile(model_path):
            device = kwargs.get("device", "cpu")
            model = get_resnet_model()
            model.load_state_dict(
                torch.load(model_path, map_location=torch.device(device))
            )
            return model
        else:
            raise FileNotFoundError(f"No such file: {model_path}")
    else:
        model = nn.Sequential(
            load_resnet(),
            nn.Linear(in_features=2_048, out_features=1_012),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=1_012, out_features=1_012),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=1_012, out_features=1),
        )
        return model


def get_siamese_model(model_path=None, **kwargs) -> Model:
    if model_path:
        if os.path.isfile(model_path):
            model = get_siamese_model(**kwargs)
            model.load_state_dict(
                torch.load(model_path, map_location=torch.device("cpu"))
            )
            return model
        else:
            raise FileNotFoundError(f"No such file: {model_path}")
    else:
        return SiameseNet(**kwargs)


def get_resnet_18(model_path: Optional[str] = None, **kwargs: int) -> Model:
    """Retrieves ResNet 18 from torchvision with modified FC layer."""
    if model_path:
        model = get_resnet_18()  # Retrieves untrained model
        if os.path.isfile(model_path):
            model.load_state_dict(
                torch.load(model_path, map_location=torch.device("cpu"))
            )
            return model
        else:
            raise FileNotFoundError(f"{model_path} not found")
    else:
        return MyResNet18(**kwargs)


def get_triplet_model(model_name: Optional[str] = None, **kwargs: int) -> Model:
    """Retrieves model defined in: 'Face Presentation Attack Detection
    in Learned Color-liked Space'"""
    if model_name:
        model = get_triplet_model(**kwargs)  # Retrieves untrained model
        file_dir = os.path.dirname(os.path.abspath(__file__))
        pre_trained_path = os.path.join(file_dir, "pre_trained")
        model_path = os.path.join(pre_trained_path, model_name)
        if os.path.isfile(model_path):
            model.load_state_dict(
                torch.load(model_path, map_location=torch.device("cpu"))
            )
            return model
        else:
            raise FileNotFoundError(f"{model_name} not found")
    else:
        return TripletNet(feature_extractor=FeatureExtractor(**kwargs))


def get_CDCN(model_path: Optional[str] = None, **kwargs: int) -> Model:
    """Retrieves CDCN from torchvision with modified FC layer."""
    if model_path:
        model = get_CDCN()  # Retrieves untrained model
        if os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path))
            return model
        else:
            raise FileNotFoundError(f"{model_path} not found")
    else:
        return CDCN(**kwargs).float()
def get_CDCN_clasifier_1(model_path: Optional[str] = None, **kwargs: int) -> Model:
    """Retrieves CDCN from torchvision with modified FC layer."""
    if model_path:
        model = get_CDCN_clasifier_1(**kwargs)  # Retrieves untrained model
        if os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path))
            return model
        else:
            raise FileNotFoundError(f"{model_path} not found")
    else:
        return CDCN_classifier_1(**kwargs)


# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         model1 = get_model(kind='resnet_50')
#         model1_1 = nn.Sequential(*list(model1.children())[:-3])
#         self.model1 = nn.Sequential(
#                 model1_1,
#                 nn.ReLU(),
#                 nn.Linear(1012, 1012))
#         # model2 = get_model(kind='vision_resnet_18')
#         self.model2 = self._base_net = models.resnet18(pretrained=True)
#         self.model2.fc = nn.Sequential(
#                 nn.Linear(in_features=512, out_features=512)
#                 )

#         self.fc2 = nn.Linear(1012+512, 1)

#     def forward(self, data):
#         x1 = self.model1(data)
#         # print(x1.shape)
#         x2 = self.model2(data)
#         # print(x2.shape)
#         x = torch.cat((x1, x2), dim=1)
#         x = self.fc2(x)
#         return x
