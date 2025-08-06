import torchvision.transforms as transforms
from torchvision.models import get_model_weights

# This file can be used to define custom data augmentations.

def get_transform(model_name):
    """
    Gets the appropriate transform for a given model.
    It uses torchvision.models.get_model_weights to get the weights and their associated transforms.
    """
    try:
        # Get the weights enum from the model name
        weights_enum = get_model_weights(model_name)
        # Get the default weights
        weights = weights_enum.DEFAULT
        # Return the transforms associated with the weights
        return weights.transforms()
    except Exception:
        # Fallback to default transformation if model-specific weights are not found
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])