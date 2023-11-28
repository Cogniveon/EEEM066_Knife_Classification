class BaseConfig(object):
    # tf_efficientnet_b0, resnet50, densenet121, vgg16_bn
    base_model = "tf_efficientnet_b0"
    ## number of classes
    n_classes = 192
    img_weight = 224
    img_height = 224
    batch_size = 32
    epochs = 20
    learning_rate = 0.005


class ResnetConfig(BaseConfig):
    base_model = "resnet18"


class VGGConfig(BaseConfig):
    base_model = "vgg13_bn"


default = BaseConfig()
resnet = ResnetConfig()
vgg = VGGConfig()
