import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import from train.py
from train import (
    Config, 
    PreprocessedLRWARDataset, 
    ArabicLipreadingModel,
    evaluate_by_tier
)
# Update these paths for your setup
CHECKPOINT_PATH = '/kaggle/input/model/best_model.pth'
DATA_PATH = '/kaggle/input/dataset-lip'
OUTPUT_DIR = '/kaggle/working'

def evaluate_model(model, loader, criterion, device):
    """Evaluate model on given dataloader"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (videos, labels) in enumerate(tqdm(loader, desc='Evaluating')):
            videos, labels = videos.to(device), labels.to(device)
            
            logits = model(videos)
            loss = criterion(logits, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Memory cleanup
            if (batch_idx + 1) % 10 == 0 and device == 'cuda':
                torch.cuda.empty_cache()
    
    loss = running_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds) * 100
    
    return loss, acc, all_preds, all_labels

def plot_training_curves(history, output_dir):
    """Plot training history"""
    if not history:
        print(" No training history available")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train', linewidth=2, color='#2E86AB')
    axes[0].plot(history['val_loss'], label='Val', linewidth=2, color='#A23B72')
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train', linewidth=2, color='#2E86AB')
    axes[1].plot(history['val_acc'], label='Val', linewidth=2, color='#A23B72')
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[2].plot(history['lr'], linewidth=2, color='#F18F01')
    axes[2].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(" Training curves saved")

def plot_confusion_matrix(labels, preds, classes, output_dir):
    """Plot confusion matrix"""
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=classes, yticklabels=classes,
        cbar_kws={'label': 'Count'}, square=True
    )
    plt.xlabel('Predicted', fontsize=13, fontweight='bold')
    plt.ylabel('True', fontsize=13, fontweight='bold')
    plt.title('Test Set Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    plt.savefig(Path(output_dir) / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(" Confusion matrix saved")

def print_per_class_analysis(labels, preds, classes, tier_mapping):
    """Print per-class accuracy breakdown"""
    print("\n" + "="*80)
    print("PER-CLASS ACCURACY ANALYSIS")
    print("="*80)
    print(f"{'Class':<15} {'Tier':<25} {'Accuracy':<12} {'Samples':<10}")
    print("-"*80)
    
    labels_np = np.array(labels)
    preds_np = np.array(preds)
    
    for i, class_name in enumerate(classes):
        mask = labels_np == i
        if mask.sum() > 0:
            acc = (preds_np[mask] == labels_np[mask]).mean() * 100
            tier = tier_mapping[class_name]
            n_samples = mask.sum()
            print(f"{class_name:<15} {tier:<25} {acc:>6.2f}%      {n_samples:>4d}")



def main():
    
    # ========== STEP 1: Load Configuration ==========
    print("\n[1/7] Loading checkpoint and configuration...")
    
    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    
    # Initialize config
    config = Config()
    config.PREPROCESSED_ROOT = DATA_PATH
    config.OUTPUT_DIR = OUTPUT_DIR
    
    # Override with saved config if available
    if 'config' in checkpoint:
        for key, value in checkpoint['config'].items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    print(f" Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"   Device: {config.DEVICE}")
    
    # ========== STEP 2: Load Test Dataset ==========
    print("\n[2/7] Loading test dataset...")
    
    test_dataset = PreprocessedLRWARDataset(
        preprocessed_root=config.PREPROCESSED_ROOT,
        split='test',
        classes=config.SELECTED_CLASSES,
        augment=False,
        config=config
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    print(f" Test set: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    # ========== STEP 3: Create Model and Load Weights ==========
    print("\n[3/7] Creating model and loading weights...")
    
    model = ArabicLipreadingModel(
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE
    ).to(config.DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(" Model loaded successfully")
    
    # ========== STEP 4: Evaluate on Test Set ==========
    print("\n[4/7] Evaluating on test set...")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    test_loss, test_acc, test_preds, test_labels = evaluate_model(
        model, test_loader, criterion, config.DEVICE
    )
    
    print(f" Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    # ========== STEP 5: Tier Analysis ==========
    print("\n[5/7] Analyzing by phonetic tier...")
    
    tier_results = evaluate_by_tier(
        test_preds, test_labels, config.SELECTED_CLASSES, config.TIER_MAPPING
    )
    
    print("\nTier Accuracy:")
    for tier_name in sorted(tier_results.keys()):
        acc = tier_results[tier_name]['accuracy']
        n = tier_results[tier_name]['n_samples']
        print(f"  {tier_name:<25}: {acc:6.2f}%  ({n:4d} samples)")
    
    # ========== STEP 6: Generate Visualizations ==========
    print("\n[6/7] Generating visualizations...")
    
    # Training curves
    if 'history' in checkpoint and checkpoint['history']:
        plot_training_curves(checkpoint['history'], config.OUTPUT_DIR)
    
    # Confusion matrix
    plot_confusion_matrix(test_labels, test_preds, config.SELECTED_CLASSES, config.OUTPUT_DIR)
    
    # ========== STEP 7: Save Results ==========
    print("\n[7/7] Saving results...")
    
    # Get validation results
    val_acc = checkpoint.get('val_acc', 0)
    val_loss = checkpoint.get('val_loss', 0)
    train_acc = checkpoint.get('train_acc', 0)
    
    # Compile results
    results = {
        'checkpoint': {
            'epoch': checkpoint['epoch'],
            'train_acc': float(train_acc) if train_acc else None,
            'val_acc': float(val_acc),
            'val_loss': float(val_loss)
        },
        'test': {
            'loss': float(test_loss),
            'accuracy': float(test_acc),
            'tier_results': {k: {
                'accuracy': float(v['accuracy']),
                'n_samples': int(v['n_samples'])
            } for k, v in tier_results.items()}
        },
        'generalization': {
            'train_val_gap': float(train_acc - val_acc) if train_acc else None,
            'val_test_gap': float(val_acc - test_acc)
        },
        'config': {
            'num_classes': config.NUM_CLASSES,
            'batch_size': config.BATCH_SIZE,
            'dropout_rate': config.DROPOUT_RATE
        }
    }
    
    # Save JSON
    with open(Path(config.OUTPUT_DIR) / 'evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(" Results saved to evaluation_results.json")
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    print(f"\nResults:")
    print(f"  Best validation accuracy: {val_acc:.2f}%")
    print(f"  Test accuracy: {test_acc:.2f}%")
    print(f"  Val-Test gap: {val_acc - test_acc:.2f}%")
    
    # Classification report
    print(classification_report(
        test_labels, test_preds,
        target_names=config.SELECTED_CLASSES,
        digits=3
    ))
    
    # Per-class analysis
    print_per_class_analysis(test_labels, test_preds, config.SELECTED_CLASSES, config.TIER_MAPPING)
    
    # Status
    val_test_gap = val_acc - test_acc
    if abs(val_test_gap) < 2:
        status = " EXCELLENT generalization"
    elif abs(val_test_gap) < 5:
        status = " GOOD generalization"
    else:
        status = " Check generalization"
    
    print(f"\n{status}")
    
    print(f"Files saved to {config.OUTPUT_DIR}/:")
    print("  - evaluation_results.json")
    print("  - training_curves.png")
    print("  - confusion_matrix.png")
    
if __name__ == "__main__":
    main()
