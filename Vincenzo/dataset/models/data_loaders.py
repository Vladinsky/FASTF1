"""
Data Loaders for LSTM Tire Change Prediction
============================================

DataLoader ottimizzati per sequenze temporali con gestione:
- Caricamento efficiente dati preprocessati (.npy files)
- Bilanciamento class imbalance con smart sampling
- Supporto CPU/GPU con pin_memory adattivo
- Augmentation per esempi positivi (minority class)
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from typing import Tuple, Dict, Optional, List
import os
from pathlib import Path
import pickle


class TireChangeSequenceDataset(Dataset):
    """
    Dataset per sequenze temporali di cambio gomme
    
    Carica i dati preprocessati e gestisce:
    - Sequenze di lunghezza fissa (10 timesteps)
    - Multi-task targets (cambio + tipo mescola)
    - Augmentation condizionale per minority class
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',  # 'train', 'val', 'test'
        augment_positive: bool = False,
        augment_factor: int = 3,
        noise_std: float = 0.01,
        device: str = 'cpu'
    ):
        """
        Args:
            data_dir: Directory con dati preprocessati
            split: Split da caricare ('train', 'val', 'test')
            augment_positive: Se True, augmenta esempi positivi
            augment_factor: Fattore moltiplicativo per augmentation
            noise_std: Deviazione standard rumore gaussiano
            device: Device target
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment_positive = augment_positive and split == 'train'
        self.augment_factor = augment_factor
        self.noise_std = noise_std
        self.device = device
        
        # Carica dati
        self._load_data()
        
        # Setup augmentation se richiesta
        if self.augment_positive:
            self._setup_augmentation()
    
    def _load_data(self):
        """Carica sequenze e targets dal disco"""
        try:
            # Carica sequenze (X)
            X_path = self.data_dir / f'X_{self.split}.npy'
            self.sequences = np.load(X_path, allow_pickle=True)
            
            # Carica targets cambio gomme (y_change)
            y_change_path = self.data_dir / f'y_change_{self.split}.npy'
            self.tire_change_targets = np.load(y_change_path, allow_pickle=True)
            
            # Carica targets tipo mescola (y_type)
            y_type_path = self.data_dir / f'y_type_{self.split}.npy'
            self.tire_type_targets = np.load(y_type_path, allow_pickle=True)
            
            print(f"✅ Loaded {self.split} data:")
            print(f"   Sequences: {self.sequences.shape}")
            print(f"   Tire change targets: {self.tire_change_targets.shape}")
            print(f"   Tire type targets: {self.tire_type_targets.shape}")
            
            # Statistiche class distribution
            positive_samples = np.sum(self.tire_change_targets)
            total_samples = len(self.tire_change_targets)
            print(f"   Positive samples: {positive_samples}/{total_samples} ({positive_samples/total_samples*100:.2f}%)")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Data files not found in {self.data_dir}. "
                                  f"Run preprocessing first. Error: {e}")
    
    def _setup_augmentation(self):
        """Setup indices per augmentation degli esempi positivi"""
        # Trova indici esempi positivi
        self.positive_indices = np.where(self.tire_change_targets == 1)[0]
        self.negative_indices = np.where(self.tire_change_targets == 0)[0]
        
        print(f"📈 Augmentation setup:")
        print(f"   Original positive samples: {len(self.positive_indices)}")
        print(f"   Augmentation factor: {self.augment_factor}")
        print(f"   Total augmented positive: {len(self.positive_indices) * self.augment_factor}")
        
        # Crea mapping per augmented indices
        self.augmented_positive_count = len(self.positive_indices) * (self.augment_factor - 1)
        self.total_samples = len(self.sequences) + self.augmented_positive_count
        
        print(f"   New dataset size: {self.total_samples} (was {len(self.sequences)})")
    
    def __len__(self) -> int:
        """Ritorna lunghezza dataset (con augmentation)"""
        if self.augment_positive:
            return self.total_samples
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Ritorna item del dataset
        
        Returns:
            Tuple (sequence, tire_change_target, tire_type_target)
        """
        original_size = len(self.sequences)
        
        if self.augment_positive and idx >= original_size:
            # Questo è un esempio augmented
            augmented_idx = idx - original_size
            positive_idx = self.positive_indices[augmented_idx % len(self.positive_indices)]
            
            # Applica augmentation con rumore
            sequence = self.sequences[positive_idx].copy()
            sequence += np.random.normal(0, self.noise_std, sequence.shape)
            
            tire_change_target = self.tire_change_targets[positive_idx]
            tire_type_target = self.tire_type_targets[positive_idx]
            
        else:
            # Esempio originale
            sequence = self.sequences[idx]
            tire_change_target = self.tire_change_targets[idx]
            tire_type_target = self.tire_type_targets[idx]
        
        # Converti a tensori
        sequence = torch.FloatTensor(sequence)
        tire_change_target = torch.LongTensor([tire_change_target])  # Binary -> LongTensor
        tire_type_target = torch.LongTensor([tire_type_target])
        
        return sequence, tire_change_target.squeeze(), tire_type_target.squeeze()
    
    def get_class_weights(self) -> torch.Tensor:
        """Calcola pesi per WeightedRandomSampler"""
        if self.augment_positive:
            # Con augmentation, ricalcola distribution
            total_positive = len(self.positive_indices) * self.augment_factor
            total_negative = len(self.negative_indices)
            total = total_positive + total_negative
            
            weight_positive = total / (2.0 * total_positive)
            weight_negative = total / (2.0 * total_negative)
        else:
            # Standard weighting
            total_positive = np.sum(self.tire_change_targets)
            total_negative = len(self.tire_change_targets) - total_positive
            total = len(self.tire_change_targets)
            
            weight_positive = total / (2.0 * total_positive)
            weight_negative = total / (2.0 * total_negative)
        
        # Crea array di pesi per ogni sample
        weights = np.zeros(len(self))
        
        if self.augment_positive:
            # Originali
            weights[:len(self.sequences)] = np.where(
                self.tire_change_targets == 1, weight_positive, weight_negative
            )
            # Augmented (tutti positivi)
            weights[len(self.sequences):] = weight_positive
        else:
            weights = np.where(
                self.tire_change_targets == 1, weight_positive, weight_negative
            )
        
        return torch.FloatTensor(weights)


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True,
    augment_positive: bool = True,
    use_weighted_sampling: bool = False,
    device: str = 'cpu'
) -> Dict[str, DataLoader]:
    """
    Crea DataLoader per tutti gli split
    
    Args:
        data_dir: Directory con dati preprocessati
        batch_size: Batch size
        num_workers: Worker per DataLoader
        pin_memory: Se True, usa pin_memory (GPU only)
        augment_positive: Se True, augmenta esempi positivi nel training
        use_weighted_sampling: Se True, usa WeightedRandomSampler
        device: Device target
        
    Returns:
        Dict con DataLoader per 'train', 'val', 'test'
    """
    
    # Adatta pin_memory per device
    pin_memory = pin_memory and device != 'cpu'
    
    # Dataset per ogni split
    train_dataset = TireChangeSequenceDataset(
        data_dir=data_dir,
        split='train',
        augment_positive=augment_positive,
        device=device
    )
    
    val_dataset = TireChangeSequenceDataset(
        data_dir=data_dir,
        split='val',
        augment_positive=False,  # No augmentation per validation
        device=device
    )
    
    test_dataset = TireChangeSequenceDataset(
        data_dir=data_dir,
        split='test',
        augment_positive=False,  # No augmentation per test
        device=device
    )
    
    # Sampler per training (opzionale)
    train_sampler = None
    if use_weighted_sampling:
        weights = train_dataset.get_class_weights()
        train_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        print(f"📊 Using WeightedRandomSampler with {len(weights)} weights")
    
    # DataLoader per training
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # No shuffle con sampler
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last per stabilità training
    )
    
    # DataLoader per validation
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    # DataLoader per test
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    print(f"✅ Created DataLoaders:")
    print(f"   Train: {len(train_loader)} batches, {len(train_dataset)} samples")
    print(f"   Val: {len(val_loader)} batches, {len(val_dataset)} samples")
    print(f"   Test: {len(test_loader)} batches, {len(test_dataset)} samples")
    print(f"   Batch size: {batch_size}, Workers: {num_workers}, Pin memory: {pin_memory}")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def check_data_distribution(data_loader: DataLoader, name: str = "DataLoader"):
    """
    Analizza distribuzione classi in un DataLoader
    
    Args:
        data_loader: DataLoader da analizzare
        name: Nome per logging
    """
    tire_change_counts = {0: 0, 1: 0}
    tire_type_counts = {}
    total_samples = 0
    
    print(f"\n📊 Analyzing {name} distribution...")
    
    for batch_idx, (sequences, tire_change_targets, tire_type_targets) in enumerate(data_loader):
        # Count tire changes
        for target in tire_change_targets:
            tire_change_counts[target.item()] += 1
        
        # Count tire types (solo per esempi positivi)
        positive_mask = tire_change_targets == 1
        if positive_mask.sum() > 0:
            for target in tire_type_targets[positive_mask]:
                tire_type_counts[target.item()] = tire_type_counts.get(target.item(), 0) + 1
        
        total_samples += len(tire_change_targets)
        
        # Log progress ogni 100 batch
        if batch_idx % 100 == 0 and batch_idx > 0:
            print(f"   Processed {batch_idx} batches...")
    
    # Report finale
    positive_ratio = tire_change_counts[1] / total_samples
    print(f"   Total samples: {total_samples}")
    print(f"   Tire changes: {tire_change_counts[1]} ({positive_ratio*100:.2f}%)")
    print(f"   No changes: {tire_change_counts[0]} ({(1-positive_ratio)*100:.2f}%)")
    print(f"   Class ratio: 1:{tire_change_counts[0]/tire_change_counts[1]:.1f}")
    
    if tire_type_counts:
        print(f"   Tire type distribution: {tire_type_counts}")


def create_cpu_optimized_loaders(data_dir: str) -> Dict[str, DataLoader]:
    """
    Crea DataLoader ottimizzati per CPU testing
    
    Args:
        data_dir: Directory dati preprocessati
        
    Returns:
        Dict con DataLoader leggeri per CPU
    """
    return create_data_loaders(
        data_dir=data_dir,
        batch_size=16,  # Ridotto per CPU
        num_workers=0,  # Single-threaded per evitare overhead
        pin_memory=False,  # CPU only
        augment_positive=True,
        use_weighted_sampling=False,  # Semplifica per test
        device='cpu'
    )


if __name__ == "__main__":
    # Test dei DataLoader
    print("Testing Tire Change DataLoaders...")
    
    # Path dati preprocessati
    data_dir = "../preprocessed"
    
    if os.path.exists(data_dir):
        # Crea DataLoader
        data_loaders = create_cpu_optimized_loaders(data_dir)
        
        # Test un batch
        train_loader = data_loaders['train']
        sequences, tire_change_targets, tire_type_targets = next(iter(train_loader))
        
        print(f"\n🧪 Test batch:")
        print(f"   Sequences shape: {sequences.shape}")
        print(f"   Tire change targets shape: {tire_change_targets.shape}")
        print(f"   Tire type targets shape: {tire_type_targets.shape}")
        print(f"   Tire change targets sample: {tire_change_targets[:5]}")
        print(f"   Tire type targets sample: {tire_type_targets[:5]}")
        
        # Analizza distribuzione
        check_data_distribution(train_loader, "Training")
        
        print("✅ DataLoader test completed successfully!")
        
    else:
        print(f"❌ Preprocessed data not found at {data_dir}")
        print("   Run data_preprocessing.py first")
