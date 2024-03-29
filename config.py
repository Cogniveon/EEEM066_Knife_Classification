import timm
from torch import nn, optim
from torchvision import models
from torchvision.models.alexnet import AlexNet_Weights
from torchvision.models.densenet import DenseNet121_Weights
from torchvision.models.googlenet import GoogLeNet_Weights
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models.shufflenetv2 import (
    ShuffleNet_V2_X1_0_Weights,
    ShuffleNet_V2_X2_0_Weights,
)
from torchvision.models.vgg import VGG11_Weights
from torchvision.models.vision_transformer import ViT_B_16_Weights


class BaseConfig(object):
    def __init__(self) -> None:
        ## number of classes
        self.n_classes = 192
        self.model_name = "tf_efficientnet_b0"
        self.base_model = timm.create_model(self.model_name, pretrained=True, num_classes=self.n_classes)
        self.img_width = 224
        self.img_height = 224
        self.batch_size = 16
        self.epochs = 20
        self.learning_rate = 0.005
        self.weight_decay = 0.001


class AlexnetConfig(BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "alexnet"
        alexnet_model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        alexnet_model.classifier[6] = nn.Linear(alexnet_model.classifier[6].in_features, self.n_classes)

        # Create a new AlexNet model and load the pre-trained weights
        self.base_model = nn.Sequential(
            alexnet_model.features, nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten(), alexnet_model.classifier
        )

        # Set requires_grad to True for fine-tuning
        for param in alexnet_model.parameters():
            param.requires_grad = True

        self.batch_size = 16
        self.learning_rate = 0.005


class EfficientB5Config(BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "tf_efficientnet_b5"
        self.base_model = timm.create_model(self.model_name, pretrained=True, num_classes=self.n_classes)
        self.batch_size = 16
        self.learning_rate = 0.005


class DensenetConfig(BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "densenet121"
        self.base_model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Linear(self.base_model.classifier.in_features, self.n_classes)
        self.batch_size = 16
        self.learning_rate = 0.05


class ResnetConfig(BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "resnet18"

        resnet_pretrained = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        sequential_layers = nn.Sequential(
            nn.Linear(resnet_pretrained.fc.in_features, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, self.n_classes),
            # nn.LogSoftmax(dim=1),
        )

        resnet_pretrained.fc = sequential_layers
        for param in resnet_pretrained.fc.parameters():
            param.requires_grad = True

        self.base_model = resnet_pretrained
        self.learning_rate = 0.001


class ShuffleNetConfig(BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "shufflenet_v2_x1_0"
        self.base_model = models.shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, self.n_classes)
        self.learning_rate = 0.005


class ShuffleNet2xConfig(BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "shufflenet_v2_x2_0"
        self.base_model = models.shufflenet_v2_x2_0(weights=ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, self.n_classes)
        self.learning_rate = 0.0005
        self.epochs = 20


class MobileNetConfig(BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "mobilenetv3_large_100"
        self.base_model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        self.base_model.classifier[-1] = nn.Linear(self.base_model.classifier[-1].in_features, self.n_classes)
        self.learning_rate = 0.005


class VITConfig(BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "vit_b_16"
        self.base_model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        # self.base_model.head = nn.Linear(self.base_model.heads.in_features, num_classes=self.n_classes)
        self.batch_size = 16
        self.learning_rate = 0.003


class GooglenetConfig(BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "googlenet"
        pretrained_googlenet = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
        pretrained_googlenet.fc = nn.Linear(pretrained_googlenet.fc.in_features, self.n_classes)

        # Create a new GoogLeNet model and load the pre-trained weights
        self.base_model = nn.Sequential(
            pretrained_googlenet.conv1,
            pretrained_googlenet.conv2,
            pretrained_googlenet.conv3,
            pretrained_googlenet.inception3a,
            pretrained_googlenet.inception3b,
            pretrained_googlenet.inception4a,
            pretrained_googlenet.inception4b,
            pretrained_googlenet.inception4c,
            pretrained_googlenet.inception4d,
            pretrained_googlenet.inception4e,
            pretrained_googlenet.inception5a,
            pretrained_googlenet.inception5b,
            pretrained_googlenet.avgpool,
            nn.Flatten(),
            pretrained_googlenet.fc,
        )

        # Set requires_grad to True for fine-tuning
        for param in self.base_model.parameters():
            param.requires_grad = True

        self.batch_size = 3
        self.learning_rate = 0.005


class TinyVITConfig(BaseConfig):
    def __init__(self) -> None:
        self.base_model = "tiny_vit_5m_224.in1k"


class VGGConfig(BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "vgg11"
        vgg11_model = models.vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
        vgg11_model.classifier[6] = nn.Linear(vgg11_model.classifier[6].in_features, self.n_classes)
        self.base_model = nn.Sequential(
            vgg11_model.features, nn.AdaptiveAvgPool2d((7, 7)), nn.Flatten(), vgg11_model.classifier
        )

        # Set requires_grad to True for fine-tuning
        for param in self.base_model.parameters():
            param.requires_grad = True

        self.batch_size = 16
        self.learning_rate = 0.005


default = BaseConfig()
efficientnetb0 = BaseConfig()
efficientnetb5 = EfficientB5Config()
alexnet = AlexnetConfig()
googlenet = GooglenetConfig()
resnet = ResnetConfig()
densenet = DensenetConfig()
shufflenet = ShuffleNetConfig()
shufflenet2x = ShuffleNet2xConfig()
mobilenet = MobileNetConfig()
tinyvit = TinyVITConfig()
vit = VITConfig()
vgg = VGGConfig()
