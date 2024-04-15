from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import torchaudio
from torchaudio.transforms import MelSpectrogram
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import pandas as pd

# Load AudioCaps dataset using Hugging Face datasets library
dataset = load_dataset("audiocaps")

# Convert dataset to Pandas DataFrame for easier manipulation
df = pd.DataFrame(dataset['train'])

# Define paths to save preprocessed datasets
output_dir = "path_to_save_datasets"
train_file = os.path.join(output_dir, "train_dataset.pth")
val_file = os.path.join(output_dir, "val_dataset.pth")
test_file = os.path.join(output_dir, "test_dataset.pth")

# Split dataset into training, validation, and testing sets
train_df, val_test_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=42)

# Function to preprocess audio and text
def preprocess_audio_text(row):
    audio, sr = torchaudio.load(row['file'])
    mel_spec_transform = MelSpectrogram(sample_rate=sr)
    mel_spec = mel_spec_transform(audio)
    
    text_tokens = word_tokenize(row['text'].lower())
    
    return mel_spec, text_tokens

# Apply preprocessing to training, validation, and testing datasets
train_dataset = train_df.apply(preprocess_audio_text, axis=1, result_type='expand')
val_dataset = val_df.apply(preprocess_audio_text, axis=1, result_type='expand')
test_dataset = test_df.apply(preprocess_audio_text, axis=1, result_type='expand')

# Save preprocessed datasets using PyTorch DataLoader
torch.save(train_dataset, train_file)
torch.save(val_dataset, val_file)
torch.save(test_dataset, test_file)
