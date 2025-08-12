import torchvision.models as models

def get_resnet(model_name='resnet50', pretrained=True):
    weights = 'IMAGENET1K_V1' if pretrained else None
    if hasattr(models, model_name):
        model = getattr(models, model_name)(weights=weights)
    else:
        # Fallback or error for models not in torchvision
        raise ValueError(f"Model {model_name} not found in torchvision.models")
    return model

resnet50 = lambda pretrained=True: get_resnet('resnet50', pretrained)
resnet18 = lambda pretrained=True: get_resnet('resnet18', pretrained)