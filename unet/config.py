import torch

config = {
    'clean_train_dir': 'dataset/split_dataset/train/',
    'clean_val_dir': 'dataset/split_dataset/test/',
    
    'batch_size': 16,
    'num_workers': 10,
    'lr': 1e-3,
    'epochs': 3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    'model_save_path': 'denoising_unet.pth'
}
