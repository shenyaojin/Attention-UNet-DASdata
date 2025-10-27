
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from unet import UNet, DiceLoss
from attention_unet import AttentionUNet
from dataset import DASDataset
from tqdm import tqdm

# Configuration
dim = (512, 512)
imgPath = 'data/img/'
lblPath = 'data/lbl/'
picPath = 'picture/'
modPath = 'model/'
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def get_model(model_path):
    """Instantiates a model based on the model path string."""
    model_name = os.path.basename(model_path)
    if 'attention_unet' in model_name:
        print("Instantiating AttentionUNet.")
        return AttentionUNet(in_ch=1, out_ch=1).to(DEVICE)
    elif 'unet' in model_name:
        print("Instantiating UNet.")
        return UNet(in_ch=1, out_ch=1).to(DEVICE)
    else:
        raise ValueError(f"Model type not recognized from path: {model_path}")

def visualize_results(model_path, data_loader, num_samples=5):
    """
    Visualizes model predictions against ground truth labels.

    Args:
        model_path (str): Path to the trained model file.
        data_loader (DataLoader): DataLoader for the dataset.
        num_samples (int): Number of samples to visualize.
    """
    print(f"Loading model from {model_path}")
    model = get_model(model_path)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['net'])
    model.eval()

    dice_loss_fn = DiceLoss()
    total_dice_score = 0
    
    model_name_str = os.path.basename(model_path).replace('_model_min_valid.pth', '')
    output_dir = os.path.join(picPath, 'visualizations', model_name_str)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        for i, (input_data, label) in enumerate(data_loader):
            if i >= num_samples:
                break
            
            input_data = input_data.to(DEVICE)
            label = label.to(DEVICE)
            
            output = model(input_data)
            
            # Calculate Dice Score
            dice_score = 1 - dice_loss_fn(output, label).item()
            total_dice_score += dice_score

            # Convert to numpy for visualization
            input_np = input_data.cpu().numpy()[0, 0]
            label_np = label.cpu().numpy()[0, 0]
            output_np = output.cpu().numpy()[0, 0]

            # Plot and save the results
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(input_np, cmap='gray')
            axes[0].set_title('Input Data')
            axes[0].axis('off')

            axes[1].imshow(label_np, cmap='gray')
            axes[1].set_title('Ground Truth Label')
            axes[1].axis('off')

            axes[2].imshow(output_np, cmap='gray')
            axes[2].set_title(f'Model Prediction\nDice: {dice_score:.4f}')
            axes[2].axis('off')

            save_path = os.path.join(output_dir, f'sample_{i}.png')
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Saved visualization for sample {i} to {save_path}")

    print(f"\nAverage Dice Score over {num_samples} samples: {total_dice_score / num_samples:.4f}")

def compare_models(model_paths, data_loader):
    """
    Compares different models based on their performance on the validation set.

    Args:
        model_paths (list): A list of paths to the trained model files.
        data_loader (DataLoader): DataLoader for the validation dataset.
    """
    results = {}
    dice_loss_fn = DiceLoss()

    for model_path in model_paths:
        print(f"Evaluating model: {os.path.basename(model_path)}")
        model = get_model(model_path)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['net'])
        model.eval()

        total_dice_score = 0
        num_batches = 0
        with torch.no_grad():
            for input_data, label in tqdm(data_loader):
                input_data = input_data.to(DEVICE)
                label = label.to(DEVICE)
                output = model(input_data)
                dice_score = 1 - dice_loss_fn(output, label).item()
                total_dice_score += dice_score
                num_batches += 1
        
        avg_dice_score = total_dice_score / num_batches
        results[os.path.basename(model_path)] = avg_dice_score

    # Print results table
    print("\n--- Model Comparison ---")
    print(f"{'Model':<40} | {'Average Dice Score':<20}")
    print("-" * 63)
    for model_name, score in sorted(results.items(), key=lambda item: item[1], reverse=True):
        print(f"{model_name:<40} | {score:<20.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    models = list(results.keys())
    scores = list(results.values())
    plt.barh([m.replace('.pth', '') for m in models], scores)
    plt.xlabel('Average Dice Score')
    plt.title('Model Performance Comparison')
    plt.tight_layout()
    save_path = os.path.join(picPath, 'model_comparison.png')
    plt.savefig(save_path)
    plt.close()
    print(f"\nSaved model comparison plot to {save_path}")


if __name__ == '__main__':
    # --- Load Data ---
    allID = [f for f in os.listdir(imgPath) if f.endswith('.dat')]
    train_len = int(len(allID) * 0.8)
    validID = allID[train_len:]
    valid_data_loader = DataLoader(DASDataset(imgPath, lblPath, validID, chann=1, dim=dim), batch_size=1, shuffle=False)

    # --- Visualize Results of Best Models ---
    best_model_files = [os.path.join(modPath, f) for f in os.listdir(modPath) if f.endswith('_min_valid.pth')]
    if best_model_files:
        for best_model_path in best_model_files:
            visualize_results(best_model_path, valid_data_loader, num_samples=5)
    else:
        print("No best models found for visualization. Please check the model directory.")

    # --- Compare All Epoch Models ---
    model_files = [os.path.join(modPath, f) for f in os.listdir(modPath) if f.endswith('.pth') and 'min' not in f]
    if model_files:
        compare_models(model_files, valid_data_loader)
    else:
        print("No models found for comparison in the model directory.")
