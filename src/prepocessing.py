import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import warnings
from tqdm import tqdm
import tempfile
import os
import re
import mediapipe as mp
warnings.filterwarnings('ignore')

class FaceMouthDetector:
    def __init__(self, use_mediapipe=True):
        self.use_mediapipe = use_mediapipe and MEDIAPIPE_AVAILABLE
        
        if self.use_mediapipe:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            print("Using MediaPipe for face detection")
        else:
            print("Using center crop fallback")
    
    def get_mouth_roi(self, frame, target_size=112, padding_ratio=0.4):
        
        if self.use_mediapipe:
            return self._get_mouth_mediapipe(frame, target_size, padding_ratio)
        else:
            return self._get_mouth_centercrop(frame, target_size)
    
    def _get_mouth_mediapipe(self, frame, target_size, padding_ratio):
        h, w = frame.shape[:2]
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face landmarks
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            # Fallback to center crop
            return self._get_mouth_centercrop(frame, target_size)
        
        # Get mouth landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Mouth region landmark indices (outer lips)
        mouth_indices = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            291, 409, 270, 269, 267, 0, 37, 39, 40, 185
        ]
        
        # Extract mouth coordinates
        mouth_points = []
        for idx in mouth_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            mouth_points.append([x, y])
        
        mouth_points = np.array(mouth_points)
        
        # Calculate bounding box
        x_min, y_min = mouth_points.min(axis=0)
        x_max, y_max = mouth_points.max(axis=0)
        
        # Add padding
        mouth_w = x_max - x_min
        mouth_h = y_max - y_min
        
        pad_w = int(mouth_w * padding_ratio)
        pad_h = int(mouth_h * padding_ratio)
        
        x1 = max(0, x_min - pad_w)
        y1 = max(0, y_min - pad_h)
        x2 = min(w, x_max + pad_w)
        y2 = min(h, y_max + pad_h)
        
        # Extract and resize
        mouth_roi = frame[y1:y2, x1:x2]
        
        if mouth_roi.size == 0:
            return self._get_mouth_centercrop(frame, target_size)
        
        # Convert to grayscale and resize
        if len(mouth_roi.shape) == 3:
            mouth_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
        
        mouth_roi = cv2.resize(mouth_roi, (target_size, target_size))
        
        return mouth_roi
    
    def _get_mouth_centercrop(self, frame, target_size):
        h, w = frame.shape[:2]
        
        # Mouth is typically in bottom 2/3 of face, centered horizontally
        center_x = w // 2
        center_y = int(h * 0.65)  # 65% down from top
        
        # Crop size (make it square)
        crop_size = min(h, w) // 2
        
        x1 = max(0, center_x - crop_size // 2)
        x2 = min(w, center_x + crop_size // 2)
        y1 = max(0, center_y - crop_size // 2)
        y2 = min(h, center_y + crop_size // 2)
        
        mouth_roi = frame[y1:y2, x1:x2]
        
        # Convert to grayscale
        if len(mouth_roi.shape) == 3:
            mouth_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize
        mouth_roi = cv2.resize(mouth_roi, (target_size, target_size))
        
        return mouth_roi


def extract_video_id(video_path):
    
    filename = Path(video_path).stem
    
    # Try to find 8-digit ID pattern (most common)
    match = re.search(r'\d{8}', filename)
    if match:
        return match.group(0)
    
    # Try to find any numeric sequence
    match = re.search(r'\d+', filename)
    if match:
        return match.group(0).zfill(8)  # Pad to 8 digits
    
    # Fallback: use full stem (without extension)
    return filename


class LRWARDataset(Dataset):

    def __init__(self, 
                 root_dir,
                 split='train',
                 classes=None,
                 num_frames=29,
                 img_size=112,
                 use_mediapipe=True,
                 augment=False,
                 preprocessed_dir=None):
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir else None
        
        # Initialize face detector (only needed if not using preprocessed)
        if self.preprocessed_dir is None:
            self.face_detector = FaceMouthDetector(use_mediapipe=use_mediapipe)
        else:
            self.face_detector = None
            print(f"✅ Loading from preprocessed directory: {self.preprocessed_dir}")
        
        # Get class names
        if classes is None:
            self.classes = sorted([d.name for d in (self.root_dir / split).iterdir() 
                                  if d.is_dir()])
        else:
            self.classes = classes
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Build sample list
        self.samples = []
        print(f"\n📂 Building {split} dataset index...")
        
        # Check if root directory exists
        split_dir = self.root_dir / split
        if not split_dir.exists():
            print(f" ERROR: Split directory does not exist: {split_dir}")
            print(f" Please check your root_dir path!")
            return
        
        for class_name in tqdm(self.classes, desc=f"Indexing {split} classes"):
            class_dir = split_dir / class_name
            
            if not class_dir.exists():
                print(f"⚠️  Warning: {class_dir} does not exist, skipping...")
                continue
            
            video_files = list(class_dir.glob('*.mp4'))
            
            if len(video_files) == 0:
                print(f"⚠️  Warning: No .mp4 files found in {class_dir}")
                continue
            
            for video_path in video_files:
                csv_path = video_path.with_suffix('.csv')
                
                # Extract video ID for safe filename
                video_id = extract_video_id(video_path)
                
                self.samples.append({
                    'video': video_path,
                    'csv': csv_path if csv_path.exists() else None,
                    'label': self.class_to_idx[class_name],
                    'class_name': class_name,
                    'split': split,
                    'video_id': video_id  # NEW: Store video ID
                })
        
        print(f"\n{split.upper()} Dataset:")
        print(f"  Classes: {len(self.classes)}")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Augmentation: {'ON' if self.augment else 'OFF'}")
        print(f"  Mode: {'PREPROCESSED' if self.preprocessed_dir else 'ON-THE-FLY'}")
        
        # Warn if no samples found
        if len(self.samples) == 0:
            print(f"\n WARNING: No samples found! Please check:")
            print(f"   1. root_dir exists: {self.root_dir.exists()}")
            print(f"   2. split directory exists: {split_dir.exists()}")
            print(f"   3. Class directories contain .mp4 files")
    
    def _get_preprocessed_path(self, sample):
       
        npy_path = (self.preprocessed_dir / 
                   sample['split'] / 
                   sample['class_name'] / 
                   f"{sample['video_id']}.npy")
        
        return npy_path
    
    def __len__(self):
        return len(self.samples)
    
    def load_video_frames(self, video_path):
        
        video_path = Path(video_path)
        
        # Read video file as binary
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
        
        # Create temp file with ASCII-safe name
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        
        try:
            # Open video from temp file (no Arabic path issues)
            cap = cv2.VideoCapture(tmp_path)
            
            # Check if video opened successfully
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path.name}")
            
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
            # Check if any frames were loaded
            if len(frames) == 0:
                raise ValueError(f"No frames could be read from video: {video_path.name}")
            
            return frames
            
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    def parse_csv_timing(self, csv_path, target_word):
        
        if csv_path is None or not csv_path.exists():
            return None
        
        try:
            # Read CSV (format: confidence, start, end, word)
            df = pd.read_csv(csv_path, header=None, 
                           names=['confidence', 'start', 'end', 'word'])
            
            # Clean word column (remove whitespace)
            df['word'] = df['word'].str.strip()
            
            # Find target word
            word_rows = df[df['word'] == target_word]
            
            if len(word_rows) > 0:
                # Get first occurrence
                row = word_rows.iloc[0]
                
                # Get start and end times
                start_time = float(row['end'])
                end_time = float(row['start'])
                
                # Ensure start < end (fix if swapped)
                if start_time > end_time:
                    start_time, end_time = end_time, start_time
                
                return (start_time, end_time)
            
        except Exception as e:
            # Silently fail and use fallback
            pass
        
        return None
    
    def temporal_alignment(self, frames, word_timing=None, fps=25):
        
        total_frames = len(frames)
        
        if total_frames == 0:
            # Emergency: return black frames
            return [np.zeros((256, 256, 3), dtype=np.uint8)] * self.num_frames
        
        # Drop first frame if we have 30 (to get odd number 29)
        if total_frames == 30:
            frames = frames[1:]
            total_frames = 29
        
        # CASE 1: Use word timing from CSV (PREFERRED METHOD)
        if word_timing is not None:
            start_time, end_time = word_timing
            
            # Convert to frame indices
            word_start_frame = int(start_time * fps)
            word_end_frame = int(end_time * fps)
            
            # Calculate center of target word
            word_center_frame = (word_start_frame + word_end_frame) // 2
            
            # Ensure center is within bounds
            word_center_frame = max(0, min(total_frames - 1, word_center_frame))
            
            # Extract 29-frame window CENTERED on word
            window_start = word_center_frame - self.num_frames // 2
            window_end = window_start + self.num_frames
            
            # Adjust if window goes out of bounds
            if window_start < 0:
                window_start = 0
                window_end = self.num_frames
            elif window_end > total_frames:
                window_end = total_frames
                window_start = max(0, window_end - self.num_frames)
            
            # Extract frames
            selected_frames = frames[window_start:window_end]
            
            # Pad if necessary
            if len(selected_frames) < self.num_frames:
                pad_before = (self.num_frames - len(selected_frames)) // 2
                pad_after = self.num_frames - len(selected_frames) - pad_before
                selected_frames = (
                    [frames[0]] * pad_before +
                    selected_frames +
                    [frames[-1]] * pad_after
                )
            
            return selected_frames[:self.num_frames]
        
        # CASE 2: No timing - use center frames (FALLBACK)
        
        # If exactly num_frames, return as is
        if total_frames == self.num_frames:
            return frames
        
        # If fewer frames, pad with edge frames
        if total_frames < self.num_frames:
            pad_before = (self.num_frames - total_frames) // 2
            pad_after = self.num_frames - total_frames - pad_before
            
            padded_frames = (
                [frames[0]] * pad_before +
                frames +
                [frames[-1]] * pad_after
            )
            return padded_frames
        
        # If more frames, center crop
        start_idx = (total_frames - self.num_frames) // 2
        return frames[start_idx:start_idx + self.num_frames]
    
    def augment_frame(self, frame):
        """Apply spatial augmentation to frame."""
        # Random brightness
        if np.random.rand() < 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            frame = np.clip(frame * alpha, 0, 255).astype(np.uint8)
        
        # Random contrast
        if np.random.rand() < 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            mean = frame.mean()
            frame = np.clip((frame - mean) * alpha + mean, 0, 255).astype(np.uint8)
        
        # Gaussian noise
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 2.5, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return frame
    
    def normalize_sequence(self, sequence):
        """
        Normalize entire sequence (per-sequence normalization).
        """
        mean = sequence.mean()
        std = sequence.std()
        
        if std < 1e-6:
            std = 1.0
        
        normalized = (sequence - mean) / std
        return normalized
    
    def process_video(self, sample):
        # Step 1: Load all video frames
        frames = self.load_video_frames(sample['video'])
        
        # Step 2: Parse CSV timing for target word
        word_timing = None
        if sample['csv'] is not None and sample['csv'].exists():
            word_timing = self.parse_csv_timing(sample['csv'], sample['class_name'])
        
        # Step 3: Temporal alignment - 29 frames CENTERED on word
        frames = self.temporal_alignment(frames, word_timing=word_timing)
        
        # Step 4: Extract mouth ROI for each frame
        mouth_rois = []
        for frame in frames:
            mouth_roi = self.face_detector.get_mouth_roi(
                frame, 
                target_size=self.img_size
            )
            mouth_rois.append(mouth_roi)
        
        # Step 5: Stack into sequence (T, H, W)
        video_sequence = np.stack(mouth_rois, axis=0).astype(np.float32)
        
        # Step 6: Normalize sequence
        video_sequence = self.normalize_sequence(video_sequence)
        
        return video_sequence
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Try loading from preprocessed .npy file
            if self.preprocessed_dir is not None:
                npy_path = self._get_preprocessed_path(sample)
                
                if npy_path.exists():
                    video_sequence = np.load(npy_path).astype(np.float32)
                else:
                    raise FileNotFoundError(f"Preprocessed file not found: {npy_path}")
            else:
                # Process on-the-fly
                video_sequence = self.process_video(sample)
            
            # Apply augmentation (training only)
            if self.augment:
                augmented_frames = []
                for frame in video_sequence:
                    frame_uint8 = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-6) * 255).astype(np.uint8)
                    frame_aug = self.augment_frame(frame_uint8)
                    augmented_frames.append(frame_aug.astype(np.float32))
                
                video_sequence = np.stack(augmented_frames, axis=0)
                video_sequence = self.normalize_sequence(video_sequence)
            
            # Convert to torch tensor (1, T, H, W)
            video_tensor = torch.from_numpy(video_sequence).unsqueeze(0)
            label = torch.tensor(sample['label'], dtype=torch.long)
            
            return video_tensor, label
        
        except Exception as e:
            print(f"Error loading {sample['video']}: {str(e)}")
            # Return dummy data
            video_tensor = torch.zeros(1, self.num_frames, self.img_size, self.img_size)
            label = torch.tensor(sample['label'], dtype=torch.long)
            return video_tensor, label


def preprocess_and_save(root_dir,
                       output_dir,
                       classes,
                       splits=['train', 'val', 'test'],
                       use_mediapipe=True):
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create error log file
    error_log_path = output_dir / "preprocessing_errors.txt"
    
    # Verify root directory exists
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"\n❌ ERROR: root_dir does not exist: {root_path}")
        return
    
    for split in splits:

        # Create dataset
        dataset = LRWARDataset(
            root_dir=root_dir,
            split=split,
            classes=classes,
            use_mediapipe=use_mediapipe,
            augment=False
        )
        
        # Process and save each video
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        for idx in tqdm(range(len(dataset)), desc=f"Preprocessing {split}"):
            sample = dataset.samples[idx]
            
            # Create directory structure
            class_output_dir = output_dir / split / sample['class_name']
            class_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get output path using video ID (NO ARABIC!)
            class_idx = dataset.class_to_idx[sample['class_name']]
            prefix = f"{class_idx:02d}" 
            npy_path = class_output_dir / f"{prefix}_{sample['video_id']}.npy"
            
            if npy_path.exists():
                skipped_count += 1
                continue
            
            try:
                # Process video
                video_sequence = dataset.process_video(sample)
                
                if video_sequence.size == 0:
                    raise ValueError("Processed video sequence is empty")
                
                # Save as .npy
                np.save(npy_path, video_sequence)
                processed_count += 1
                
            except Exception as e:
                error_count += 1
                error_msg = f"{sample['video']}: {str(e)}\n"
                
                with open(error_log_path, 'a', encoding='utf-8') as f:
                    f.write(error_msg)
                
                if error_count <= 10:
                    print(f"\n Error: {sample['video'].name} - {str(e)[:50]}")
                continue
        
        print(f" {split.upper()} preprocessing complete!")
        print(f"   Processed: {processed_count} videos")
        print(f"   Skipped: {skipped_count} videos")
        print(f"   Errors: {error_count} videos")
        if error_count > 0:
            print(f"   ⚠️  See {error_log_path}")
    
    print(f"\n{'='*70}")
    print(f"ALL PREPROCESSING COMPLETE!")
    print(f"✅ Saved to: {output_dir}")
    print(f"✅ Filenames: VIDEO_ID.npy (e.g., 00000563.npy)")
    print(f"{'='*70}\n")


def create_dataloaders(root_dir, 
                       classes,
                       batch_size=16,
                       num_workers=4,
                       use_mediapipe=True,
                       preprocessed_dir=None):
    
    # Create datasets
    train_dataset = LRWARDataset(
        root_dir=root_dir,
        split='train',
        classes=classes,
        use_mediapipe=use_mediapipe,
        augment=True,
        preprocessed_dir=preprocessed_dir
    )
    
    val_dataset = LRWARDataset(
        root_dir=root_dir,
        split='val',
        classes=classes,
        use_mediapipe=use_mediapipe,
        augment=False,
        preprocessed_dir=preprocessed_dir
    )
    
    test_dataset = LRWARDataset(
        root_dir=root_dir,
        split='test',
        classes=classes,
        use_mediapipe=use_mediapipe,
        augment=False,
        preprocessed_dir=preprocessed_dir
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n✅ Dataloaders created!")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    print("="*70 + "\n")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Selected 20 classes (phonetically diverse)
    SELECTED_CLASSES = [
        # Labials (6)
        'من', 'في', 'باسم', 'بعد', 'اليوم', 'الدولة',

        # Dentals (5)
        'الذي', 'تنظيم', 'الرئيس', 'السعودية', 'علي',

        # Emphatics (5)
        'صالح', 'طائرات', 'شخصا', 'شمال', 'الموجز',

        # Pharyngeals (4)
        'عن', 'عدد', 'عليكم', 'الحكومة',

        # Velars (3)
        'قناة', 'قطر', 'قتل'
    ]
    
    # IMPORTANT: USE THE CORRECT PATH TO YOUR DATASET!
    # Change this to match YOUR actual dataset location
    DATASET_ROOT = r'C:\Users\Azouz\Downloads\LRW-AR'
    PREPROCESSED_ROOT = r'D:\prerocessed4'
    
    # STEP 1: Preprocess and save all data
    print("="*70)
    print("STEP 1: PREPROCESSING ALL DATA")
    print("="*70)
    
    preprocess_and_save(
        root_dir=DATASET_ROOT,
        output_dir=PREPROCESSED_ROOT,
        classes=SELECTED_CLASSES,
        splits=['train', 'val', 'test'],
        use_mediapipe=MEDIAPIPE_AVAILABLE
    )
    
    # STEP 2: Test loading from preprocessed files
    dataset = LRWARDataset(
        root_dir=DATASET_ROOT,
        split='train',
        classes=SELECTED_CLASSES,
        preprocessed_dir=PREPROCESSED_ROOT,
        augment=True
        )
    
    # Check if dataset has samples
    if len(dataset) == 0:
        print("\n ERROR: No samples loaded!")
        print("Please verify:")
        print(f"  1. Dataset path: {DATASET_ROOT}")
    else:
        # Test loading one sample
        print(f"\n Loading sample...")
        video, label = dataset[0]
        
        print(f" Sample loaded successfully!")
        print(f"   Video shape: {video.shape}")  # Should be (1, 29, 112, 112)
        print(f"   Label: {label.item()}")
        print(f"   Class: {dataset.classes[label.item()]}")
        
        # Show preprocessed file path
        sample = dataset.samples[0]
        npy_path = dataset._get_preprocessed_path(sample)
        print(f"  Loaded from: {npy_path}")
        