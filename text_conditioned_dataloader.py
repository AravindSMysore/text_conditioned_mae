import os
import json
import random
from typing import List, Tuple, Set

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torchvision.transforms as transforms

# Set device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize tokenizer and text encoder once to avoid repeated loading
tokenizer_path = 'Alibaba-NLP/gte-large-en-v1.5'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
text_encoder = AutoModel.from_pretrained(tokenizer_path, trust_remote_code=True).to(device)
text_encoder.eval()

class ImageTextDataset(Dataset):
    def __init__(
        self,
        image_caption_pairs: List[Tuple[str, List[str]]],
        all_captions: Set[str],
        positive_pair_prob: float = 0.5
    ):
        self.image_caption_pairs = image_caption_pairs
        self.all_captions = list(all_captions)
        self.positive_pair_prob = positive_pair_prob

        # Image preprocessing transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, torch.Tensor]:
        image_path, captions = self.image_caption_pairs[idx]
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        
        # Decide if this will be a positive or negative pair
        is_positive = random.random() < self.positive_pair_prob
        
        if is_positive:
            caption = random.choice(captions)
            label = torch.tensor(1)
        else:
            caption = random.choice([cap for cap in self.all_captions if cap not in captions])
            label = torch.tensor(0)
        
        return image_tensor, caption, label

def get_text_embeddings(texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        batch_dict = tokenizer(
            texts,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        outputs = text_encoder(**batch_dict)
        embeddings = outputs.last_hidden_state.detach().cpu()
        text_attention_mask = batch_dict['attention_mask'].detach().cpu()
        
    return embeddings, text_attention_mask

def collate_fn(batch):
    images, captions, labels = zip(*batch)

    # Stack images into a single tensor (B, C, H, W)
    images_tensor = torch.stack(images)

    # Get text embeddings and attention masks for the batch of captions
    embeddings, text_attention_mask = get_text_embeddings(captions)

    # Stack labels into a single tensor (B,)
    labels_tensor = torch.stack(labels)

    return images_tensor.to(device), embeddings.to(device), text_attention_mask.bool().to(device), labels_tensor.to(device)

def create_dataloaders(
    train_data_path: str,
    val_data_path: str,
    train_image_folder: str,
    val_image_folder: str,
    batch_size: int = 32,
    negative_prob: float = 0.5
) -> Tuple[DataLoader, DataLoader]:
    
    def extract_images_and_captions(image_folder: str, data: dict) -> Tuple[List[Tuple[str, List[str]]], Set[str]]:
        images_dict = {img['id']: img['file_name'] for img in data['images']}
        captions_dict = {}
        all_captions_set = set()
        
        for ann in data['annotations']:
            img_id = ann['image_id']
            caption = ann['caption'].strip()
            all_captions_set.add(caption)
            captions_dict.setdefault(img_id, []).append(caption)
        
        pairs_list = [(os.path.join(image_folder, images_dict[img_id]), captions_dict[img_id]) 
                      for img_id in images_dict if img_id in captions_dict]
        
        return pairs_list, all_captions_set

    with open(train_data_path, 'r') as file:
        train_data = json.load(file)

    with open(val_data_path, 'r') as file:
        val_data = json.load(file)

    train_image_caption_pairs, train_all_captions = extract_images_and_captions(train_image_folder, train_data)
    val_image_caption_pairs, val_all_captions = extract_images_and_captions(val_image_folder, val_data)

    train_dataset = ImageTextDataset(train_image_caption_pairs, train_all_captions, positive_pair_prob=negative_prob)
    val_dataset = ImageTextDataset(val_image_caption_pairs, val_all_captions, positive_pair_prob=negative_prob)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    return train_loader, val_loader

# Example usage
def get_dataloaders(batch_size: int = 32, negative_prob: float = 0.5) -> Tuple[DataLoader, DataLoader]:
    train_file_path = "C:\\Users\\aravi\\OneDrive\\Aravind Important\\Carnegie Mellon\\Semesters\\Fall24\\MML\\Project\\Dataset\\annotations\\captions_train2017.json"
    val_file_path = "C:\\Users\\aravi\\OneDrive\\Aravind Important\\Carnegie Mellon\\Semesters\\Fall24\\MML\\Project\\Dataset\\annotations\\captions_val2017.json"
    train_image_folder = "C:\\Users\\aravi\\OneDrive\\Aravind Important\\Carnegie Mellon\\Semesters\\Fall24\\MML\\Project\\Dataset\\train2017"
    val_image_folder = "C:\\Users\\aravi\\OneDrive\\Aravind Important\\Carnegie Mellon\\Semesters\\Fall24\\MML\\Project\\Dataset\\val2017"

    train_loader, val_loader = create_dataloaders(
        train_file_path,
        val_file_path,
        train_image_folder,
        val_image_folder,
        batch_size=batch_size,
        negative_prob=negative_prob
    )
    return train_loader, val_loader