
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class Config: 
    # Paths
    PREPROCESSED_ROOT = '/kaggle/input/dataset-lip'
    OUTPUT_DIR = '/kaggle/working'
    
    # Classes
    NUM_CLASSES = 23
    SELECTED_CLASSES = [
        'من', 'في', 'باسم', 'بعد', 'اليوم', 'الدولة',
        'الذي', 'تنظيم', 'الرئيس', 'السعودية', 'علي',
        'صالح', 'طائرات', 'شخصا', 'شمال', 'الموجز',
        'عن', 'عدد', 'عليكم', 'الحكومة',
        'قناة', 'قطر', 'قتل'
    ]
    
    TIER_MAPPING = {
        'من': 'TIER1_LABIALS', 'في': 'TIER1_LABIALS', 'باسم': 'TIER1_LABIALS',
        'بعد': 'TIER1_LABIALS', 'اليوم': 'TIER1_LABIALS', 'الدولة': 'TIER1_LABIALS',
        'الذي': 'TIER2_DENTALS', 'تنظيم': 'TIER2_DENTALS', 'الرئيس': 'TIER2_DENTALS',
        'السعودية': 'TIER2_DENTALS', 'علي': 'TIER2_DENTALS',
        'صالح': 'TIER3_EMPHATICS', 'طائرات': 'TIER3_EMPHATICS', 'شخصا': 'TIER3_EMPHATICS',
        'شمال': 'TIER3_EMPHATICS', 'الموجز': 'TIER3_EMPHATICS',
        'عن': 'TIER4_PHARYNGEALS', 'عدد': 'TIER4_PHARYNGEALS',
        'عليكم': 'TIER4_PHARYNGEALS', 'الحكومة': 'TIER4_PHARYNGEALS',
        'قناة': 'TIER5_VELARS', 'قطر': 'TIER5_VELARS', 'قتل': 'TIER5_VELARS',
    }
    
    # Training settings
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.02
    DROPOUT_RATE = 0.35
    LABEL_SMOOTHING = 0.05
    MIXUP_ALPHA = 0.2
    
    # Learning rate schedule
    LR_SCHEDULER = 'cosine'
    WARMUP_EPOCHS = 5
    MIN_LR = 1e-6
    PATIENCE = 20
    
    # System
    NUM_WORKERS = 2
    PIN_MEMORY = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 42

class PreprocessedLRWARDataset(Dataset):

    def __init__(self, preprocessed_root, split='train', classes=None, augment=False, config=None):
        self.preprocessed_root = Path(preprocessed_root)
        self.split = split
        self.augment = augment and (split == 'train')
        self.config = config
        
        if classes is None:
            self.classes = sorted([d.name for d in (self.preprocessed_root / split).iterdir() 
                                  if d.is_dir()])
        else:
            self.classes = classes
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        split_dir = self.preprocessed_root / split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        for class_name in tqdm(self.classes, desc=f"Indexing {split}", leave=False):
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            
            npy_files = list(class_dir.glob('*.npy'))
            for npy_path in npy_files:
                self.samples.append({
                    'npy_path': npy_path,
                    'label': self.class_to_idx[class_name],
                    'class_name': class_name
                })
        
        print(f"{split.upper()}: {len(self.samples)} samples")
    
    def time_masking(self, sequence):
        """Time masking augmentation"""
        T = sequence.shape[0]
        if np.random.rand() < 0.5:
            mask_len = np.random.randint(1, min(5, max(2, T // 6)))
            start = np.random.randint(0, T - mask_len + 1)
            mean_frame = sequence.mean(axis=0, keepdims=True)
            sequence[start:start + mask_len] = mean_frame
        return sequence
    
    def horizontal_flip(self, sequence):
        """Horizontal flip augmentation"""
        if np.random.rand() < 0.5:
            sequence = np.flip(sequence, axis=2).copy()
        return sequence
    
    def augment_sequence(self, sequence):
        """Apply augmentations"""
        sequence = self.horizontal_flip(sequence)
        sequence = self.time_masking(sequence)
        return sequence
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            video_sequence = np.load(sample['npy_path']).astype(np.float32)
            
            if video_sequence.max() > 1.0:
                video_sequence = video_sequence / 255.0
            
            if self.augment:
                video_sequence = self.augment_sequence(video_sequence)
            
            video_tensor = torch.from_numpy(video_sequence).unsqueeze(0)
            label = torch.tensor(sample['label'], dtype=torch.long)
            return video_tensor, label
        except Exception as e:
            print(f"Error loading {sample['npy_path']}: {e}")
            video_tensor = torch.zeros(1, 29, 112, 112)
            label = torch.tensor(sample['label'], dtype=torch.long)
            return video_tensor, label



class SpatioTemporalEmbedding3D(nn.Module):
    """3D CNN frontend"""
    def __init__(self, in_channels=1, out_channels=64, dropout=0.15):
        super().__init__()
        self.conv3d_1 = nn.Conv3d(in_channels, 32, (3,5,5), (1,1,1), (1,2,2), bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout3d(dropout)
        
        self.conv3d_2 = nn.Conv3d(32, out_channels, (3,3,3), (1,1,1), (1,1,1), bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout3d(dropout)
    
    def forward(self, x):
        x = self.dropout1(self.relu1(self.bn1(self.conv3d_1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.conv3d_2(x))))
        return x

class ChannelAttention(nn.Module):
    """Channel attention mechanism"""
    def __init__(self, in_channels, reduction=16, dropout=0.1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, T, C = x.shape
        x_reshaped = x.view(B*T, C, 1)
        avg_out = self.avg_pool(x_reshaped).view(B*T, C)
        weights = self.fc(avg_out).view(B, T, C)
        return x * weights

class TemporalAttention(nn.Module):
    """Temporal attention mechanism"""
    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        attn_scores = self.attention(x)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(x * attn_weights, dim=1)
        return context, attn_weights.squeeze(-1)

class ArabicLipreadingModel(nn.Module):
    """Complete Arabic lipreading model"""
    def __init__(self, num_classes=23, dropout_rate=0.35):
        super().__init__()
        
        self.num_classes = num_classes
        self.conv3d_channels = 64
        self.hidden_dim = 256
        
        # 3D CNN frontend
        self.spatiotemporal_embed = SpatioTemporalEmbedding3D(
            1, self.conv3d_channels, dropout=dropout_rate * 0.4
        )
        
        # Skip connection
        self.skip_pool = nn.AdaptiveAvgPool3d((29, 1, 1))
        self.skip_projection = nn.Sequential(
            nn.Linear(self.conv3d_channels, 576),
            nn.Dropout(dropout_rate * 0.3)
        )
        
        # MobileNetV3-Small backbone
        backbone = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        backbone.features[0][0] = nn.Conv2d(
            self.conv3d_channels, 16, 3, 2, 1, bias=False
        )
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.eff_dim = 576
        
        # Channel attention
        self.channel_attention = ChannelAttention(
            self.eff_dim, 16, dropout=dropout_rate * 0.3
        )
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            self.eff_dim, self.hidden_dim, 1,
            batch_first=True, bidirectional=True, dropout=0
        )
        self.lstm_out_dim = self.hidden_dim * 2
        self.lstm_dropout = nn.Dropout(dropout_rate)
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(
            self.lstm_out_dim, dropout=dropout_rate * 0.5
        )
        
        # Classifier
        fusion_input_dim = self.lstm_out_dim + self.eff_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(fusion_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(512, num_classes)
        )
        
        self._init_weights()
        
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params: {total:,} (~{total*4/1024/1024:.1f}MB)")
        print(f"Trainable params: {trainable:,} (~{trainable*4/1024/1024:.1f}MB)")
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        
        # 3D CNN
        spatio_feat = self.spatiotemporal_embed(x)
        
        # Skip connection
        skip = self.skip_pool(spatio_feat).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        skip = self.skip_projection(skip)
        
        # Extract per-frame features
        eff_list = []
        for t in range(T):
            feat = self.backbone(spatio_feat[:, :, t, :, :]).view(B, -1)
            eff_list.append(feat)
        eff_feat = torch.stack(eff_list, dim=1)
        
        # Channel attention
        eff_feat = self.channel_attention(eff_feat)
        
        # Add skip connection
        eff_feat = eff_feat + skip
        
        # LSTM
        lstm_out, _ = self.lstm(eff_feat)
        lstm_out = self.lstm_dropout(lstm_out)
        
        # Temporal attention
        temp_ctx, temp_attn = self.temporal_attention(lstm_out)
        
        # Attended features
        eff_ctx = torch.sum(eff_feat * temp_attn.unsqueeze(-1), dim=1)
        
        # Fusion and classification
        combined = torch.cat([temp_ctx, eff_ctx], dim=1)
        logits = self.classifier(combined)
        
        return logits


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_lr_multiplier(epoch, warmup_epochs, total_epochs):
    """Cosine learning rate schedule with warmup"""
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    else:
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

def train_one_epoch(model, loader, criterion, optimizer, device, config, use_mixup=True):
    """Train for one epoch with gradient accumulation"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    accumulation_steps = 2
    
    pbar = tqdm(loader, desc='Training', leave=False)
    optimizer.zero_grad()
    
    for batch_idx, (videos, labels) in enumerate(pbar):
        videos, labels = videos.to(device), labels.to(device)
        
        # Apply mixup
        if use_mixup and np.random.rand() < 0.5:
            videos, labels_a, labels_b, lam = mixup_data(videos, labels, config.MIXUP_ALPHA)
            logits = model(videos)
            loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
        else:
            logits = model(videos)
            loss = criterion(logits, labels)
        
        # Gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # Metrics
        running_loss += loss.item() * accumulation_steps
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item() * accumulation_steps, 'acc': 100 * correct / total})
        
        if (batch_idx + 1) % 10 == 0 and device == 'cuda':
            torch.cuda.empty_cache()
    
    # Handle remaining gradients
    if (batch_idx + 1) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    return running_loss / len(loader), 100 * correct / total

def validate(model, loader, criterion, device):
    """Validate the model"""
    from sklearn.metrics import accuracy_score
    
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (videos, labels) in enumerate(tqdm(loader, desc='Validating', leave=False)):
            videos, labels = videos.to(device), labels.to(device)
            
            logits = model(videos)
            loss = criterion(logits, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0 and device == 'cuda':
                torch.cuda.empty_cache()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds) * 100
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def evaluate_by_tier(preds, labels, classes, tier_mapping):
    """Evaluate accuracy by phonetic tier"""
    from sklearn.metrics import accuracy_score
    
    label_to_class = {idx: cls for idx, cls in enumerate(classes)}
    label_to_tier = {idx: tier_mapping[label_to_class[idx]] for idx in range(len(classes))}
    
    tier_results = {}
    for tier_name in set(tier_mapping.values()):
        tier_idx = [i for i, lbl in enumerate(labels) if label_to_tier[lbl] == tier_name]
        if len(tier_idx) == 0:
            continue
        
        tier_preds = [preds[i] for i in tier_idx]
        tier_labels = [labels[i] for i in tier_idx]
        tier_acc = accuracy_score(tier_labels, tier_preds) * 100
        tier_results[tier_name] = {'accuracy': tier_acc, 'n_samples': len(tier_idx)}
    
    return tier_results



def main():
    """Main training function"""
    # Initialize config
    config = Config()
    
    # Set random seeds
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"\n✅ Configuration:")
    print(f"   Device: {config.DEVICE}")
    print(f"   Batch size: {config.BATCH_SIZE}")
    print(f"   Epochs: {config.NUM_EPOCHS}")
    print(f"   Learning rate: {config.LEARNING_RATE}")
    print(f"   Dropout: {config.DROPOUT_RATE}")
    
    # Create datasets and dataloaders
    train_dataset = PreprocessedLRWARDataset(
        preprocessed_root=config.PREPROCESSED_ROOT,
        split='train',
        classes=config.SELECTED_CLASSES,
        augment=True,
        config=config
    )
    
    val_dataset = PreprocessedLRWARDataset(
        preprocessed_root=config.PREPROCESSED_ROOT,
        split='val',
        classes=config.SELECTED_CLASSES,
        augment=False,
        config=config
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Create model
    
    if config.DEVICE == 'cuda':
        torch.cuda.empty_cache()
    
    model = ArabicLipreadingModel(
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE
    ).to(config.DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Training loop
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    
    start_time = time.time()
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        print("-" * 70)
        
        # Adjust learning rate
        lr_mult = get_lr_multiplier(epoch - 1, config.WARMUP_EPOCHS, config.NUM_EPOCHS)
        current_lr = max(config.LEARNING_RATE * lr_mult, config.MIN_LR)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, config, use_mixup=True
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, config.DEVICE
        )
        
        # Tier results
        tier_results = evaluate_by_tier(
            val_preds, val_labels, config.SELECTED_CLASSES, config.TIER_MAPPING
        )
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Print results
        print(f"LR: {current_lr:.6f}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        gap = train_acc - val_acc
        print(f"Train-Val Gap: {gap:.2f}%")
        
        # Save best model
        if epoch < 20:
            is_best = val_loss < best_val_loss
        else:
            is_best = val_acc > best_val_acc and val_loss < best_val_loss * 1.1
        
        if is_best:
            best_val_loss = min(val_loss, best_val_loss)
            best_val_acc = max(val_acc, best_val_acc)
            patience_counter = 0
            
            save_path = Path(config.OUTPUT_DIR) / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'config': vars(config),
                'history': history
            }, save_path)
            
            print(f"✅ New best! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter}/{config.PATIENCE} epochs")
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\n⚠️ Early stopping triggered")
            break
        
        if config.DEVICE == 'cuda':
            torch.cuda.empty_cache()
    
    training_time = time.time() - start_time
    
    
    print("TRAINING COMPLETE!")
    print(f"Total time: {training_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {config.OUTPUT_DIR}/best_model.pth")

if __name__ == "__main__":
    main()