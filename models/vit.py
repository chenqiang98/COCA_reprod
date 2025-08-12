import torchvision.models as models
import timm

def get_vit(model_name='vit_b_16', pretrained=True):
    weights = 'IMAGENET1K_V1' if pretrained else None
    if hasattr(models, model_name):
        model = getattr(models, model_name)(weights=weights)
    else:
        raise ValueError(f"Model {model_name} not found in torchvision.models")
    return model

def get_mobilevit(model_name='mobilevit_s', pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained)
    return model

vit_base_patch16_224 = lambda pretrained=True: get_vit('vit_b_16', pretrained)
mobilevit_s = lambda pretrained=True: get_mobilevit('mobilevit_s', pretrained)
