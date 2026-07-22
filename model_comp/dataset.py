import os
import re

import numpy as np
import pandas as pd
import torch


def extract_label(filename):
    """Extract class label from filename"""
    match = re.match(r'^[^_#\d]+', filename)
    return match.group(0) if match else None


def load_deep_learning_data(data_dir, file_list, sensor_list, window_size, stride,
                           start_row=600, end_row=3001, augmentation=None,
                           is_train=False, augment_factor=2):
    """Load data for deep learning models (time series format)"""

    all_data = []
    all_labels = []
    # labels = ['Onion', 'Incense', 'Sanitizer', 'Vicks', 'Camphor', 'Pepper', 'Background', 'Clinic', 'Turmeric', 'Coriander', 'Soap', 'Chocolate', 'Red', 'Garlic', 'Cardamom', 'Volini', 'Cumin', 'Ginger', 'Polo', 'Honey']
    # target_classes= {'Onion', 'Clinic', 'Vicks', 'Polo', 'Turmeric', 'Cardamom'}

    # target_classes = {'Vicks','Polo','Honey','Onion','clinic'}
    # target_classes = {'Sanitizer', 'Camphor',  'Polo', 'Ginger', 'Garlic' }#, 'Pepper', 'Turmeric', 'Vicks', 'Soap', 'Cumin', 'Incense', 'Clinic', 'Onion'}
    print(f"Loading deep learning data: sensors={sensor_list}, window={window_size}, stride={stride}")

    for file_name in file_list:
        label = extract_label(file_name)
        # if label is not in target class then continue
        # if label  in target_classes:
        #     # print(label)
        #     continue
        # if label is None:
        #     continue

        file_path = os.path.join(data_dir, file_name)

        try:
            # Load CSV
            df = pd.read_csv(file_path)

            # Check if all sensors exist
            missing_sensors = [s for s in sensor_list if s not in df.columns]
            if missing_sensors:
                print(f"Warning: {missing_sensors} not found in {file_name}")
                continue

            # Get multi-sensor data
            sensor_data = df[sensor_list].iloc[start_row:end_row].values

            # Basic outlier removal for each sensor
            for i, sensor in enumerate(sensor_list):
                col_data = sensor_data[:, i]
                rolling_mean = pd.Series(col_data).rolling(window=5, center=True).mean()
                rolling_std = pd.Series(col_data).rolling(window=5, center=True).std()
                z_scores = (pd.Series(col_data) - rolling_mean) / rolling_std

                sensor_data[:, i] = np.where(
                    np.abs(z_scores) < 2,
                    col_data,
                    rolling_mean.fillna(method='ffill').fillna(method='bfill')
                )

            # Apply smoothing to each sensor
            for i in range(len(sensor_list)):
                sensor_data[:, i] = pd.Series(sensor_data[:, i]).rolling(
                    window=7, center=True
                ).mean().fillna(method='ffill').fillna(method='bfill').values

            # Create sliding windows (keep as time series for deep learning)
            max_length = len(sensor_data)

            if max_length < window_size:
                continue

            for start in range(0, max_length - window_size + 1, stride):
                window = sensor_data[start:start + window_size]  # Shape: (window_size, n_sensors)
                all_data.append(window)
                all_labels.append(label)

                 # Data augmentation for training
                if is_train and augmentation is not None:
                    for _ in range(augment_factor):
                        aug_window = augmentation.augment(window)
                        all_data.append(aug_window)
                        all_labels.append(label)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue

    return np.array(all_data), np.array(all_labels)


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
