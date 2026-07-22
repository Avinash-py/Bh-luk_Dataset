# Enhanced Data Augmentation
import numpy as np


class AdvancedAugmentation:
    def __init__(self, noise_std=0.01, time_warp_sigma=0.2, magnitude_warp_sigma=0.2):
        self.noise_std = noise_std
        self.time_warp_sigma = time_warp_sigma
        self.magnitude_warp_sigma = magnitude_warp_sigma
    
    def add_noise(self, data):
        """Add Gaussian noise"""
        noise = np.random.normal(0, self.noise_std, data.shape)
        return data + noise
    
    def time_warp(self, data):
        """Time warping augmentation"""
        seq_len, n_features = data.shape
        time_steps = np.arange(seq_len)
        
        # Create warping
        warped_steps = time_steps + np.random.normal(0, self.time_warp_sigma, seq_len)
        warped_steps = np.sort(np.clip(warped_steps, 0, seq_len - 1))
        
        # Interpolate
        warped_data = np.zeros_like(data)
        for i in range(n_features):
            warped_data[:, i] = np.interp(time_steps, warped_steps, data[:, i])
        
        return warped_data
    
    def magnitude_warp(self, data):
        """Magnitude warping"""
        seq_len, n_features = data.shape
        warp_factors = np.random.normal(1, self.magnitude_warp_sigma, (seq_len, 1))
        warp_factors = np.clip(warp_factors, 0.5, 1.5)
        return data * warp_factors
    
    def window_slice(self, data, slice_ratio=0.9):
        """Random window slicing"""
        seq_len, n_features = data.shape
        slice_len = int(seq_len * slice_ratio)
        start_idx = np.random.randint(0, seq_len - slice_len + 1)
        
        sliced_data = data[start_idx:start_idx + slice_len]
        
        # Pad to original length
        if sliced_data.shape[0] < seq_len:
            padding = np.zeros((seq_len - sliced_data.shape[0], n_features))
            sliced_data = np.vstack([sliced_data, padding])
        
        return sliced_data
    
    def augment(self, data, augment_prob=0.8):
        """Apply random augmentations"""
        if np.random.random() > augment_prob:
            return data
        
        augmented = data.copy()
        
        # Apply random combination of augmentations
        if np.random.random() < 0.15: # 0.3:
            augmented = self.add_noise(augmented)
        if np.random.random() < 0.15: #0.3:
            augmented = self.time_warp(augmented)
        if np.random.random() < 0.15: #0.3:
            augmented = self.magnitude_warp(augmented)
        if np.random.random() < 0.2:
            augmented = self.window_slice(augmented)
        
        return augmented