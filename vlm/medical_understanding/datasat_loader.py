import torch
import torch.nn as nn
import numpy as np
import csv
import json
import torch.utils.data as data
import os
from pathlib import Path
from collections import Counter
import dask.dataframe as dd

path_xray_images = Path(r"D:\chestxray_dataset\resized_images\256")
path_xray_labels = Path(r"D:\chestxray_dataset\cleaned_dataset.csv")


def split_dataset(csv_path:Path,train_ration:float=0.8, seed:int=42):
    """
    Split the dataset into training and testing sets based on the provided ratio.

    Args:
        csv_path (Path): Path to the CSV file containing the dataset.
        train_ratio (float): Ratio of the dataset to be used for training (default is 0.8).
        seed (int): Random seed for reproducibility (default is 42)."""
    '''folder = csv.reader(open(csv_path, 'r'))
    next(folder)  # Skip the header row
    data_list = list(folder)
    projections = [row['projection'] for row in data_list]
    projection_counts = Counter(projections)
    n_frontal,n_lateral = projection_counts['Frontal'],projection_counts['Lateral']
    frontals = [row for row in data_list if row['projection'] == 'Frontal']
    laterals = [row for row in data_list if row['projection'] == 'Lateral']
    np.random.seed(seed)
    np.random.shuffle(frontals)
    np.random.shuffle(laterals)'''
    df = dd.read_csv(csv_path)
    df_frontal = df[df['projection'] == 'Frontal']
    df_lateral = df[df['projection'] == 'Lateral']
    df_frontal_train, df_frontal_test = df_frontal.random_split([train_ration, 1 - train_ration], random_state=seed)
    df_lateral_train, df_lateral_test = df_lateral.random_split([train_ration, 1 - train_ration], random_state=seed)
    df_train = dd.concat([df_frontal_train, df_lateral_train])
    df_test = dd.concat([df_frontal_test, df_lateral_test])
    path_train_csv = Path(r"D:\chestxray_dataset\train_dataset.csv")
    path_test_csv = Path(r"D:\chestxray_dataset\test_dataset.csv")
    dd.to_csv(df_train, path_train_csv, single_file=True,index=False)
    dd.to_csv(df_test, path_test_csv, single_file=True, index=False)

class ChestXrayDataset(data.Dataset):
    def __init__(self,csv_path:str, image_size:int=224, mode:str='train',transform=None):
        self.csv_path = Path(csv_path)
        self.image_size = image_size
        self.transform = transform
        self.mode = mode
        self.data = []
        self.create_dataset()
    def create_dataset(self):
        with open(self.csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                image_id = row['image_id']
                view = row['projection']
                output_label = row['org_caption']
                self.data.append((image_id, view, output_label))
    def prepare_message(self, image_id:str, view:str, output_label:str):
        image_path = path_xray_images / image_id
        if not image_path.exists():
            raise FileNotFoundError(f"Image file {image_path} does not exist.")
        system_prompt = "You are to act as a radiologist and write the finding section of a chest x-ray radiology report for this X-ray image. Write in the style of a radiologist, if any anomalies or alarming signs are detected, provide a detailed description."
        input_prompt = f"This is a {view} view of a chest x-ray, provide a clear medical diagnosis, if any anomalies or alarming signs are detected, provide a detailed description."
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": input_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": output_label}],
            },
        ]

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        image_id, view, output_label = self.data[idx]
        message = self.prepare_message(image_id, view, output_label)
        return message


