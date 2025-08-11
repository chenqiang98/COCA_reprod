import argparse
import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
import json


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Evaluate models on ImageNet-C')

    # 必需参数
    parser.add_argument('--data_dir', default='data/ImageNet-C', type=str, required=False,
                        help='Path to ImageNet-C dataset directory')
    parser.add_argument('--corruption', default='gaussian_noise', type=str, required=False,
                        help='Corruption type (e.g. gaussian_noise) or "all" for all types')
    parser.add_argument('--severity', default=5, type=int, required=False, choices=range(1, 6),
                        help='Severity level (1-5)')
    parser.add_argument('--all_severities', action='store_true',
                        help='Test all severity levels (1-5)')

    # 模型选择
    parser.add_argument('--resnet', action='store_true',
                        help='Evaluate ResNet50 model')
    parser.add_argument('--vit', action='store_true',
                        help='Evaluate ViT model')

    # 可选参数
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation (default: 32)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use (default: auto)')

    return parser.parse_args()


def get_device(device_pref):
    """根据偏好获取设备"""
    if device_pref == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device('cpu')


def get_all_corruption_types():
    """获取所有可用的corruption types"""
    # ImageNet-C的标准corruption types
    corruption_types = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
        'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    return corruption_types


def load_imagenet_labels():
    """加载ImageNet标签映射"""
    try:
        with open('data/imagenet_class_index.json', 'r') as f:
            imagenet_labels = json.load(f)
        return imagenet_labels
    except FileNotFoundError:
        print("Warning: imagenet_class_index.json not found, using default labels")
        return None


def prepare_resnet():
    """准备ResNet50模型和预处理"""
    model = models.resnet50(pretrained=True)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return model, transform


def prepare_vit():
    """准备ViT模型和预处理 - 使用timm的vit_base_patch16_224"""
    
    # 使用timm的vit_base_patch16_224模型
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1000)
    
    # 使用标准的ImageNet预处理，与ResNet保持一致
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return model, transform


def evaluate_model(model, dataloader, device, model_name, processor=None):
    """评估模型性能"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"Evaluating {model_name}")):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 每100个batch打印一次进度和当前精度
            if (batch_idx + 1) % 100 == 0:
                current_accuracy = 100 * correct / total
                print(f"\n[{model_name}] Batch {batch_idx + 1}/{len(dataloader)} - "
                      f"Current Accuracy: {current_accuracy:.2f}% "
                      f"({correct}/{total} correct)")

    accuracy = 100 * correct / total
    print(f"\n{model_name} Final Accuracy: {accuracy:.2f}%")
    return accuracy


def test_single_corruption(model, processor, transform, data_root, corruption, severity, 
                          batch_size, device, model_name):
    """测试单个corruption type"""
    data_path = os.path.join(data_root, corruption, str(severity))
    
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} does not exist, skipping...")
        return None
    
    try:
        dataset = datasets.ImageFolder(root=data_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, num_workers=4)
        
        print(f"Dataset size: {len(dataset)} images")
        print(f"Number of batches: {len(dataloader)}")
        print(f"Batch size: {batch_size}")
        print(f"Number of classes: {len(dataset.classes)}")
        
        # 显示前几个类别名称
        if len(dataset.classes) > 0:
            print(f"First few classes: {dataset.classes[:5]}")
        
        acc = evaluate_model(model, dataloader, device, model_name, processor)
        return acc
    except Exception as e:
        print(f"Error testing {corruption} severity {severity}: {e}")
        return None


def main():
    args = parse_args()

    # 设置设备
    device = get_device(args.device)
    print(f"Using device: {device}")

    # 加载ImageNet标签映射
    imagenet_labels = load_imagenet_labels()
    if imagenet_labels:
        print(f"Loaded ImageNet labels: {len(imagenet_labels)} classes")

    # 确定要测试的corruption types
    if args.corruption == 'all':
        corruption_types = get_all_corruption_types()
        print(f"Testing all {len(corruption_types)} corruption types")
    else:
        corruption_types = [args.corruption]
        print(f"Testing corruption type: {args.corruption}")

    # 确定要测试的severity levels
    if args.all_severities:
        severity_levels = list(range(1, 6))
        print(f"Testing severity levels: {severity_levels}")
    else:
        severity_levels = [args.severity]
        print(f"Testing severity level: {args.severity}")

    results = {}

    # 准备ResNet50
    if args.resnet:
        print("Preparing ResNet50...")
        model, transform = prepare_resnet()
        model = model.to(device)
        results['ResNet50'] = {}

        for corruption in corruption_types:
            results['ResNet50'][corruption] = {}
            for severity in severity_levels:
                print(f"\nTesting ResNet50 on {corruption} severity {severity}...")
                acc = test_single_corruption(model, None, transform, args.data_dir, 
                                           corruption, severity, args.batch_size, device, 'ResNet50')
                if acc is not None:
                    results['ResNet50'][corruption][severity] = acc
                    print(f"ResNet50 on {corruption} severity {severity}: {acc:.2f}%")

    # 准备ViT
    if args.vit:
        print("Preparing ViT...")
        model, transform = prepare_vit()
        model = model.to(device)
        results['ViT'] = {}

        for corruption in corruption_types:
            results['ViT'][corruption] = {}
            for severity in severity_levels:
                print(f"\nTesting ViT on {corruption} severity {severity}...")
                acc = test_single_corruption(model, None, transform, args.data_dir, 
                                           corruption, severity, args.batch_size, device, 'ViT')
                if acc is not None:
                    results['ViT'][corruption][severity] = acc
                    print(f"ViT on {corruption} severity {severity}: {acc:.2f}%")

    # 打印结果摘要
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        print("-" * 30)
        
        for corruption in corruption_types:
            if corruption in model_results:
                print(f"\n{corruption}:")
                for severity in severity_levels:
                    if severity in model_results[corruption]:
                        acc = model_results[corruption][severity]
                        print(f"  Severity {severity}: {acc:.2f}%")
                
                # 计算该corruption的平均准确率
                valid_accs = [model_results[corruption][s] for s in severity_levels 
                             if s in model_results[corruption] and model_results[corruption][s] is not None]
                if valid_accs:
                    avg_acc = sum(valid_accs) / len(valid_accs)
                    print(f"  Average: {avg_acc:.2f}%")

    # 保存结果到文件
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    filename = f"source_only_test_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    # 准备保存的数据
    save_data = {
        'timestamp': timestamp,
        'test_config': {
            'data_dir': args.data_dir,
            'corruption_types': corruption_types,
            'severity_levels': severity_levels,
            'batch_size': args.batch_size,
            'device': str(device)
        },
        'results': results
    }
    
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    main()