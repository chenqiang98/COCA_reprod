import timm

def get_vit(model_name='vit_base_patch16_224', pretrained=True):
    # Use timm to load ViT-Base to avoid torchvision availability issues
    model = timm.create_model(model_name, pretrained=pretrained)
    return model

def get_mobilevit(model_name='mobilevit_s', pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained)
    return model

vit_base_patch16_224 = lambda pretrained=True: get_vit('vit_base_patch16_224', pretrained)
mobilevit_s = lambda pretrained=True: get_mobilevit('mobilevit_s', pretrained)
