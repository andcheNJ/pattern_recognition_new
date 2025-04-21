import torch
import os
import random
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from multiprocessing import freeze_support # <--- IMPORT THIS

# Import from your scripts folder
from scripts.model import SiameseNetwork
from scripts.loss import ContrastiveLoss
from scripts.utils import SiameseDataset, calculate_mean_std, visualize_embeddings, SimpleImageDataset


# --- Place ALL your execution code inside a function or directly in the if block ---
# --- It's often cleaner in a function: ---

def run_training():
    # --- Configuration ---
    dataset_path = r'C:\Users\Andrew\OneDrive\Dokumente\Arduino\image_project\data\mini_sample' # Use your actual path
    epochs = 50  # Increased epochs, rely on early stopping
    batch_size = 32
    initial_lr = 0.0001 # Start with a smaller learning rate
    weight_decay = 1e-5 # Regularization
    margin = 1.0 # Contrastive loss margin (tune if needed)
    random_seed = 42 # For reproducibility
    patience = 10 # Increased patience for early stopping
    visualization_frequency = 5 # How often to run t-SNE (epochs)

    # Define model save paths
    output_dir = r'C:\Users\Andrew\OneDrive\Dokumente\Arduino\image_project\models'
    best_model_path = os.path.join(output_dir, 'siamese_best_model.pth')
    final_model_path = os.path.join(output_dir, 'siamese_final_model.pth')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # --- Reproducibility ---
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        # Might make things slower, but increases reproducibility
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Calculate Dataset Statistics ---
    print("Calculating dataset mean and std...")
    # Ensure calculate_mean_std uses a temporary dataset or handles pairs correctly
    try:
        # Run calculation with num_workers=0 temporarily if needed inside the func
        # to avoid recursive spawning during calculation itself.
        # Or ensure calculate_mean_std is safe to be imported multiple times.
        # Let's assume calculate_mean_std is safe for now or handles its own loader correctly.
         dataset_mean, dataset_std = calculate_mean_std(dataset_path)
    except Exception as e:
        # Check if error is the same multiprocessing error, if so, maybe force num_workers=0 here
        print(f"Error calculating mean/std: {e}. Using default values (0.5, 0.5).")
        dataset_mean, dataset_std = 0.5, 0.5 # Fallback values

    print(f"Dataset Mean: {dataset_mean:.4f}, Std: {dataset_std:.4f}")


    # --- Transforms ---
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[dataset_mean], std=[dataset_std])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[dataset_mean], std=[dataset_std])
    ])

    # --- Dataset & DataLoaders ---
    print("Loading dataset...")
    full_dataset = SiameseDataset(dataset_path, transform=None)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    if train_size == 0 or val_size == 0:
         raise ValueError("Dataset split resulted in 0 samples for train or validation.")

    print(f"Splitting dataset: Train={train_size}, Validation={val_size}")
    generator = torch.Generator().manual_seed(random_seed)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform

    # Determine num_workers based on OS/CPU
    # IMPORTANT: If you still have issues, try setting num_workers=0 first!
    num_workers = 4 # Your original value
    # num_workers = 0 # <-- SET TO 0 FOR DEBUGGING MULTIPROCESSING ISSUES
    print(f"Using {num_workers} workers for DataLoaders.")

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False,
        drop_last=True,
        # Add persistent_workers=True if using PyTorch 1.7+ and num_workers > 0
        # persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False,
        # persistent_workers=True if num_workers > 0 else False
    )

    # --- Create a SEPARATE DataLoader for Visualization ---
    print("Creating visualization dataset...")
    vis_dataset = SimpleImageDataset(dataset_path, transform=val_transform)
    vis_subset = torch.utils.data.Subset(vis_dataset, val_subset.indices)

    vis_loader = DataLoader(
        vis_subset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0, # Usually safe to keep visualization loader at 0 workers
        pin_memory=True if device == 'cuda' else False
    )
    print("Datasets and DataLoaders ready.")

    # --- Model Setup ---
    print("Initializing model...")
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss(margin=margin)
    optimizer = Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=initial_lr / 100)

    # --- Training Loop ---
    best_val_loss = float('inf')
    patience_counter = 0

    print("Starting training...")
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        batch_count = 0
        # The error happens when this loop starts trying to get data from workers
        for i, (img1, img2, pair_labels) in enumerate(train_loader):
            batch_count += 1
            img1 = img1.to(device, non_blocking=True).float()
            img2 = img2.to(device, non_blocking=True).float()
            pair_labels = pair_labels.to(device, non_blocking=True).float()

            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, pair_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_train_loss += loss.item()
            if (i + 1) % 50 == 0:
                 print(f"  Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}, Train Loss: {loss.item():.4f}", end='\r')

        avg_train_loss = running_train_loss / batch_count if batch_count > 0 else 0.0

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        batch_count = 0
        with torch.no_grad():
            for img1, img2, pair_labels in val_loader:
                batch_count += 1
                img1 = img1.to(device, non_blocking=True).float()
                img2 = img2.to(device, non_blocking=True).float()
                pair_labels = pair_labels.to(device, non_blocking=True).float()
                output1, output2 = model(img1, img2)
                loss = criterion(output1, output2, pair_labels)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / batch_count if batch_count > 0 else 0.0

        # --- Epoch End ---
        current_lr = optimizer.param_groups[0]['lr']
        print()
        print(f"Epoch {epoch+1}/{epochs} Summary:")
        print(f"  Avg Train Loss: {avg_train_loss:.4f}")
        print(f"  Avg Val Loss:   {avg_val_loss:.4f}")
        print(f"  Current LR:     {current_lr:.6f}")

        scheduler.step()

        # --- Save Best Model & Early Stopping ---
        if avg_val_loss < best_val_loss:
            print(f"  Validation loss improved ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving best model...")
            best_val_loss = avg_val_loss
            patience_counter = 0
            try:
                torch.save(model.state_dict(), best_model_path)
            except Exception as e:
                print(f"Error saving model: {e}")
        else:
            patience_counter += 1
            print(f"  Validation loss did not improve. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # --- Periodic Visualization ---
        if (epoch + 1) % visualization_frequency == 0 or epoch == epochs - 1:
            print("Running visualization...")
            try:
                 visualize_embeddings(model, vis_loader, device, epoch + 1)
            except Exception as e:
                 print(f"Error during visualization: {e}")


    # --- Post-Training ---
    print("\nTraining finished.")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        try:
            model.load_state_dict(torch.load(best_model_path))
        except Exception as e:
            print(f"Error loading best model weights: {e}. Proceeding with the final model state.")
    else:
        print("No best model was saved.")

    print(f"Saving final model state to {final_model_path}")
    try:
        torch.save(model.state_dict(), final_model_path)
    except Exception as e:
        print(f"Error saving final model: {e}")

    print("Script completed.")


# --- Main execution guard ---
if __name__ == '__main__':
    freeze_support()  # <--- ADD THIS LINE FIRST
    run_training()    # <--- CALL YOUR MAIN LOGIC FUNCTION