class DefaultConfig(object):
    # tf_efficientnet_b0, resnet50, densenet121, vgg16_bn
    base_model = "vgg16_bn"
    ## number of classes
    n_classes = 192
    img_weight = 224
    img_height = 224
    batch_size = 16
    epochs = 20
    learning_rate = 0.00005


config = DefaultConfig()
