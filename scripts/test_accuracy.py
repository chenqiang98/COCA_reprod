import torch
import torch.nn.functional as F
from data.imagenet_c import ImageNetC
from utils.augmentations import get_transform
from tqdm import tqdm

def test_accuracy(model, data_root, batch_size, workers, corruption, severity, anchor_model_name='vit_base_patch16_224', aux_model_name='resnet50', shuffle=False, n_examples=None):
    """
    Calculates accuracies on a given ImageNet-C corruption for:
    - anchor model
    - auxiliary model
    - combined COCA prediction (ensemble with learned tau)

    Returns a dict with keys: anchor, aux, combined
    """
    transform_anchor = get_transform(anchor_model_name)
    transform_aux = get_transform(aux_model_name)

    dataset = ImageNetC(root=data_root, corruption_type=corruption, severity=severity,
                        transform_anchor=transform_anchor, transform_aux=transform_aux, n_examples=n_examples)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=shuffle)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.anchor_model.eval()
    model.aux_model.eval()

    total = 0
    anchor_correct = 0
    aux_correct = 0
    comb_correct = 0

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating", leave=False)
        for images_anchor, images_aux, labels in pbar:
            images_anchor = images_anchor.to(device)
            images_aux = images_aux.to(device)
            labels = labels.to(device)

            p_a = model.anchor_model(images_anchor)
            p_s = model.aux_model(images_aux)

            # anchor
            anchor_pred = p_a.argmax(dim=1)
            anchor_correct += (anchor_pred == labels).sum().item()

            # aux
            aux_pred = p_s.argmax(dim=1)
            aux_correct += (aux_pred == labels).sum().item()

            # combined: follow COCA forward logic with current tau
            p_e_prime = p_a + p_s / model.tau.detach()
            
            # adaptive balance factor T with numerical stability
            max_p_e_prime = torch.max(p_e_prime, dim=1, keepdim=True)[0]
            max_p_a = torch.max(p_a, dim=1, keepdim=True)[0]
            T = max_p_e_prime / torch.clamp(max_p_a, min=1e-8)
            p_e = p_e_prime / torch.clamp(T, min=1e-8)

            comb_pred = p_e.argmax(dim=1)
            comb_correct += (comb_pred == labels).sum().item()

            total += labels.size(0)

    if total == 0:
        return {'anchor': 0, 'aux': 0, 'combined': 0}

    return {
        'anchor': (anchor_correct / total) * 100,
        'aux': (aux_correct / total) * 100,
        'combined': (comb_correct / total) * 100
    }
