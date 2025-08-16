import argparse
import torch
import torch.optim as optim
from torchvision import transforms
import json
import os
from datetime import datetime
import yaml
from tqdm import tqdm
from models.coca import COCA, get_model
from data.imagenet_c import ImageNetC
from scripts.test_accuracy import test_accuracy
from utils.augmentations import get_transform

def main():
    parser = argparse.ArgumentParser(description='COCA Test-Time Adaptation')
    parser.add_argument('--config', type=str, default='configs/vit_base_resnet50.yaml', help='Path to config file')
    parser.add_argument('--data_root', type=str, default=None, help='Path to ImageNet-C dataset')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training and testing')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--corruption', type=str, default='gaussian_noise', help='Type of corruption to test, or "all" to test all')
    parser.add_argument('--severity', type=int, default=5, help='Severity of corruption')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    args = parser.parse_args()

    # Load config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override args with config values if not provided in command line
    if args.data_root is None:
        args.data_root = config['dataset']['path']
    if args.batch_size is None:
        args.batch_size = config['dataset']['batch_size']
    args.debug = config['debug'] if 'debug' in config else False

    if args.corruption == 'all':
        # Support two ImageNet-C layouts
        # 1) Category layout: <root>/noise/gaussian_noise/5
        # 2) Flat layout:     <root>/gaussian_noise/5
        top_dirs = [d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]
        known_cats = {'noise', 'blur', 'weather', 'digital'}
        # list of known corruption names from the paper
        known_corrs = {
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
            'snow', 'frost', 'fog', 'brightness',
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        }
        # Detect category layout
        if any(d in known_cats for d in top_dirs):
            corruption_types = []
            for cat in sorted(top_dirs):
                if cat not in known_cats:
                    continue
                cat_dir = os.path.join(args.data_root, cat)
                for corr in sorted(os.listdir(cat_dir)):
                    if corr in known_corrs and os.path.isdir(os.path.join(cat_dir, corr)):
                        corruption_types.append(os.path.join(cat, corr))
        else:
            # Flat layout: use top-level dirs that match known corruptions
            corruption_types = [d for d in sorted(top_dirs) if d in known_corrs]
    else:
        # Allow either plain corruption name or nested category/corruption
        if os.path.sep in args.corruption:
            corruption_types = [args.corruption]
        else:
            # Try to find category folder containing this corruption
            found = None
            for cat in os.listdir(args.data_root):
                cat_dir = os.path.join(args.data_root, cat)
                if os.path.isdir(os.path.join(cat_dir, args.corruption)):
                    found = os.path.join(cat, args.corruption)
                    break
            corruption_types = [found or args.corruption]

    results = {}
    for corruption_type in corruption_types:
        print(f"--- Testing corruption: {corruption_type} severity: {args.severity} ---")
        metrics = run_test(args, config, corruption_type)
        results[corruption_type] = metrics

    save_results(args, config, results)

def run_test(args, config, corruption_type):
    anchor_model_config = config['model']['large_model']
    aux_model_config = config['model']['small_model']
    
    lr_anchor = anchor_model_config.get('lr', 0.001) # Default lr
    lr_aux = aux_model_config.get('lr', 0.00025) # Default lr

    # Load models
    anchor_model_name = anchor_model_config['name']
    aux_model_name = aux_model_config['name']
    anchor_model = get_model(anchor_model_name, pretrained=anchor_model_config['pretrained'])
    aux_model = get_model(aux_model_name, pretrained=aux_model_config['pretrained'])

    # Setup COCA
    # Norm adaptation config (follow Tent: BN-only by default; allow LayerNorm for ViTs if enabled)
    coca_cfg = config.get('coca', {})
    include_bn = coca_cfg.get('include_batchnorm', True)
    include_ln = coca_cfg.get('include_layernorm', False)
    include_gn = coca_cfg.get('include_groupnorm', False)
    include_in = coca_cfg.get('include_instancenorm', False)

    coca = COCA(
        anchor_model,
        aux_model,
        lr_anchor=lr_anchor,
        lr_aux=lr_aux,
        momentum=args.momentum,
        include_batchnorm=include_bn,
        include_layernorm=include_ln,
        include_groupnorm=include_gn,
        include_instancenorm=include_in,
    )

    # Data loading
    transform_anchor = get_transform(anchor_model_name)
    transform_aux = get_transform(aux_model_name)
    
    dataset = ImageNetC(root=args.data_root, corruption_type=corruption_type, severity=args.severity,
                        transform_anchor=transform_anchor, transform_aux=transform_aux)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

    # Training loop (Test-Time Adaptation)
    for images_anchor, images_aux, _ in tqdm(data_loader, desc=f"Adapting on {corruption_type}", leave=False):
        if torch.cuda.is_available():
            images_anchor = images_anchor.cuda()
            images_aux = images_aux.cuda()
        
        coca.update(images_anchor, images_aux, debug=args.debug)

    # Evaluation
    accs = test_accuracy(coca, args.data_root, args.batch_size, args.workers, corruption_type, args.severity,
                         anchor_model_name=anchor_model_name, aux_model_name=aux_model_name)
    print(f"Accuracies on {corruption_type} (sev {args.severity}) -> anchor: {accs['anchor']:.2f}% | aux: {accs['aux']:.2f}% | combined: {accs['combined']:.2f}%")
    return accs

def save_results(args, config, results):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    anchor_model_name = config['model']['large_model']['name']
    aux_model_name = config['model']['small_model']['name']
    lr_anchor = config['model']['large_model'].get('lr', 0.001)
    lr_aux = config['model']['small_model'].get('lr', 0.00025)

    result_data = {
        'timestamp': timestamp,
        'models': {
            'anchor': anchor_model_name,
            'auxiliary': aux_model_name,
        },
        'dataset': config['dataset']['name'],
        'severity': args.severity,
        'hyperparameters': {
            'lr_anchor': lr_anchor,
            'lr_aux': lr_aux,
            'momentum': args.momentum,
            'batch_size': args.batch_size,
        },
        'results': {corr: {k: f"{v:.2f}%" for k, v in metrics.items()} for corr, metrics in results.items()}
    }

    if len(results) > 1:
        # compute averages across corruptions
        sums = {'anchor': 0.0, 'aux': 0.0, 'combined': 0.0}
        for metrics in results.values():
            for k in sums:
                sums[k] += metrics[k]
        avgs = {k: f"{(sums[k] / max(len(results),1)):.2f}%" for k in sums}
        result_data['average_accuracy'] = avgs
        corruption_name = "all_corruptions"
    else:
        corruption_name = args.corruption

    filename = f"result_{timestamp}_{anchor_model_name}_{aux_model_name}_{corruption_name}_sev{args.severity}.json"
    filepath = os.path.join(results_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(result_data, f, indent=4)
    
    print(f"Results saved to {filepath}")


if __name__ == '__main__':
    main()
