import argparse
import yaml
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.imagenet_c import get_imagenet_c_loader
from models.vit import get_vit
from models.resnet import get_resnet
from utils.metrics import accuracy
from datetime import datetime

def test_single_model_accuracy():
    parser = argparse.ArgumentParser(description='Single Model Accuracy Test')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create model from the 'large_model' configuration
    model_config = config['model']['large_model']
    if 'vit' in model_config['name']:
        model = get_vit(model_config['name'], model_config['pretrained'])
    else:
        model = get_resnet(model_config['name'], model_config['pretrained'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Define corruption categories
    corruption_categories = {
        'noise': ['gaussian_noise', 'shot_noise', 'impulse_noise'],
        'blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
        'weather': ['frost', 'snow', 'fog', 'brightness'],
        'digital': ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    }
    all_corruptions = [corr for cat in corruption_categories.values() for corr in cat]
    # severities = [1, 2, 3, 4, 5]
    severities = [5] # exclusively test severity 5

    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_fname = f"results/{model_config['name']}_{timestamp}_single_model_accuracy.txt"

    category_accuracies = {cat: [] for cat in corruption_categories.keys()}

    for corruption in all_corruptions:
        for severity in severities:
            print(f"Testing on {corruption}, severity {severity}")
            loader = get_imagenet_c_loader(config['dataset']['path'], corruption, severity,
                                            config['dataset']['batch_size'],
                                            num_workers=4,
                                            anchor_model_name=model_config['name'],
                                            aux_model_name=None)

            total_acc = 0
            with torch.no_grad():
                for i, (images, labels) in enumerate(loader):
                    # print(images.shape, labels)
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    acc = accuracy(logits, labels)
                    total_acc += acc
            
            avg_acc = total_acc / len(loader)
            print(f"Accuracy: {avg_acc}")
            
            with open(result_fname, 'a') as f:
                f.write(f"{corruption}, {severity}, {avg_acc}\n")

            for category, corruptions_in_category in corruption_categories.items():
                if corruption in corruptions_in_category:
                    category_accuracies[category].append(avg_acc)

    print("\n--- Corruption Category Accuracies ---")
    for category, accuracies in category_accuracies.items():
        if accuracies:
            avg_cat_acc = sum(accuracies) / len(accuracies)
            print(f"{category.capitalize()} Accuracy: {avg_cat_acc}")
            with open(result_fname, 'a') as f:
                f.write(f"{category.capitalize()} Accuracy: {avg_cat_acc}\n")
        else:
            print(f"{category.capitalize()} Accuracy: No data")
            with open(result_fname, 'a') as f:
                f.write(f"{category.capitalize()} Accuracy: No data\n")
    print(f"Results saved to {result_fname}")
    return result_fname

if __name__ == '__main__':
    test_single_model_accuracy()
