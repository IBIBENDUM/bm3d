import torch

config = {
    'noisy_train_dir': 'dataset/noisy_dataset/noiseVariance20/train/',
    'clean_train_dir': 'dataset/clean_dataset/train/',
    'noisy_val_dir': 'dataset/noisy_dataset/noiseVariance20/test/',
    'clean_val_dir': 'dataset/clean_dataset/test/',
    
    'batch_size': 16,
    'num_workers': 10,
    'lr': 1e-3,
    'epochs': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    'model_save_path': 'denoising_unet.pth'
}
