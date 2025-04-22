# --- utils.py (Revised SiameseDataset part) ---
import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random # Add import for random choices

class SiameseDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform

        # List ONLY directories (ignore files)
        self.classes = sorted([d.name for d in os.scandir(dataset_path) if d.is_dir()]) # Sort for consistency
        if not self.classes:
            raise ValueError(f"No class subdirectories found in {dataset_path}")

        self.image_paths = []
        self.labels = []
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}

        # --- Store indices for each class ---
        self.class_to_indices = {idx: [] for idx in range(len(self.classes))}
        # ----------------------------------

        # Load images from each class directory
        current_index = 0
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.isdir(class_path):
                continue

            for img_name in sorted(os.listdir(class_path)): # Sort for consistency
                img_path = os.path.join(class_path, img_name)
                # Basic check for image files
                if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)
                    self.class_to_indices[class_idx].append(current_index) # Store index for this class
                    current_index += 1
                # else:
                #     print(f"Warning: Skipping non-image or invalid file: {img_path}")

        if not self.image_paths:
             raise RuntimeError(f"No valid image files found in subdirectories of {dataset_path}")

        print(f"Found {len(self.image_paths)} images in {len(self.classes)} classes.")
        # Optional: Print class counts
        # for idx, indices in self.class_to_indices.items():
        #    print(f"  Class '{self.idx_to_class[idx]}' ({idx}): {len(indices)} images")


    def __len__(self):
        """Returns the total number of anchor samples"""
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Returns a pair of images and a label indicating similarity.
        Label = 0 if images are from the same class (positive pair)
        Label = 1 if images are from different classes (negative pair)
        (This matches the ContrastiveLoss implementation provided where label=1 means dissimilar)
        """
        try:
            # --- Anchor Image ---
            anchor_path = self.image_paths[index]
            anchor_label = self.labels[index]
            anchor_img = Image.open(anchor_path).convert('L') # Ensure grayscale

            # --- Select Pair Image ---
            # Decide if positive or negative pair (50% chance approx.)
            should_get_positive = random.choice([True, False])

            if should_get_positive and len(self.class_to_indices[anchor_label]) > 1:
                # --- Positive Pair ---
                possible_indices = self.class_to_indices[anchor_label]
                # Filter out the anchor index itself
                filtered_indices = [i for i in possible_indices if i != index]
                if not filtered_indices: # Should only happen if class has 1 image, handled by outer 'if'
                     # Fallback: Use anchor itself if absolutely necessary (should be rare)
                     pair_index = index
                else:
                     pair_index = random.choice(filtered_indices)
                pair_label = 0 # Same class

            else:
                # --- Negative Pair ---
                # Find labels different from the anchor label
                negative_labels = [l for l in self.class_to_indices.keys() if l != anchor_label]

                if not negative_labels:
                     # Edge case: Only one class exists in the dataset. Cannot form true negative pairs.
                     # Option 1: Raise error
                     # raise RuntimeError("Cannot form negative pairs: Only one class found.")
                     # Option 2: Return a positive pair instead (less ideal for training)
                     print(f"Warning: Only one class ({anchor_label}) found. Returning a positive pair instead of negative.")
                     possible_indices = self.class_to_indices[anchor_label]
                     filtered_indices = [i for i in possible_indices if i != index]
                     if not filtered_indices: pair_index = index
                     else: pair_index = random.choice(filtered_indices)
                     pair_label = 0 # Treat as positive pair
                     # Option 3: Return duplicate anchor with negative label (least ideal)
                     # pair_index = index
                     # pair_label = 1 # Treat as negative pair

                else:
                    # Select a random different class
                    negative_label = random.choice(negative_labels)
                    # Select a random image index from that class
                    pair_index = random.choice(self.class_to_indices[negative_label])
                    pair_label = 1 # Different class

            pair_path = self.image_paths[pair_index]
            pair_img = Image.open(pair_path).convert('L')

            # --- Apply Transformations ---
            if self.transform:
                anchor_img = self.transform(anchor_img)
                pair_img = self.transform(pair_img)

            return anchor_img, pair_img, torch.tensor(pair_label, dtype=torch.float) # Use float for loss

        except Exception as e:
            print(f"Error processing index {index} (anchor: {self.image_paths[index]}): {str(e)}")
            # Return None or dummy data might cause issues in DataLoader collation
            # It's often better to let the error propagate or handle it carefully
            # For simplicity here, we raise it again, but you might want robust error handling
            raise e



class SimpleImageDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_paths = []
        self.labels = [] # <--- ADD LABELS LIST

        # --- ADD class discovery and mapping ---
        self.classes = sorted([d.name for d in os.scandir(dataset_path) if d.is_dir()])
        if not self.classes:
            raise ValueError(f"No class subdirectories found in {dataset_path} for SimpleImageDataset")
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        # ----------------------------------------

        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name] # <--- Get class index
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.isdir(class_path): continue
            for img_name in sorted(os.listdir(class_path)):
                img_path = os.path.join(class_path, img_name)
                if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx) # <--- STORE THE LABEL

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        try:
            img_path = self.image_paths[index]
            label = self.labels[index] # <--- GET THE LABEL
            img = Image.open(img_path).convert('L')
            if self.transform:
                img = self.transform(img)
            # Return both image and label
            return img, torch.tensor(label, dtype=torch.long) # <--- RETURN IMG AND LABEL (use long tensor for labels)
        except Exception as e:
             print(f"Error loading image {img_path} or label for SimpleImageDataset: {e}")
             # Handle error appropriately, maybe return None and filter in DataLoader?
             # For now, re-raise
             raise e

# --- calculate_mean_std function remains the same (it uses SimpleImageDataset correctly) ---

# --- visualize_embeddings function remains the same (it expects (img, label) which it will now get) ---

# --- calculate_mean_std remains the same, but uses the ORIGINAL __getitem__ ---
# Temporarily modify or create a separate dataset class for calculation if needed,
# OR modify calculate_mean_std to handle the tuple output (ignore pair_img and label)
def calculate_mean_std(dataset_path):
    # Define a dataset class specifically for mean/std calculation


    calc_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        # transforms.Grayscale(), # Already converted in __getitem__
        transforms.ToTensor()
    ])

    dataset = SimpleImageDataset(dataset_path, transform=calc_transform)
    if len(dataset) == 0:
        raise ValueError("Dataset for mean/std calculation is empty. Check dataset_path.")

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2) # Increase batch size for speed

    mean = 0.0
    std = 0.0
    n_pixels = 0

    # More accurate calculation: sum pixel values and squared values
    sum_pixels = 0.0
    sum_sq_pixels = 0.0
    total_pixels = 0

    for images in loader: # Only images are returned now
        # images shape: [batch, channels, height, width]
        batch_pixels = images.numel() # Total pixels in the batch
        sum_pixels += torch.sum(images)
        sum_sq_pixels += torch.sum(images**2)
        total_pixels += batch_pixels

    if total_pixels == 0:
         raise ValueError("No pixels found in dataset for mean/std calculation.")

    mean = sum_pixels / total_pixels
    # Var = E[X^2] - (E[X])^2
    variance = (sum_sq_pixels / total_pixels) - (mean ** 2)
    # Clamp variance to avoid sqrt of negative due to floating point errors
    std = torch.sqrt(torch.clamp(variance, min=1e-6))

    print(f"Calculated Mean: {mean.item()}, Std: {std.item()}")
    return mean.item(), std.item()


# --- visualize_embeddings remains largely the same ---
# Ensure the dataloader passed uses a dataset that returns (image, label)
# or adapt the loop to handle (img1, img2, label) and only use img1
def visualize_embeddings(model, dataloader, device, epoch_num): # Pass device and epoch
    model.eval()
    embeddings = []
    true_labels = [] # Store original class labels, not pair labels

    # We need the base dataset to get original labels if dataloader uses SiameseDataset
    # It's cleaner if the dataloader for visualization uses a simple (img, label) dataset
    # Assuming here dataloader provides (img, label) for simplicity:
    # You might need to create a separate val_loader with a SimpleImageDataset for this.

    # --- Alternative if dataloader returns (img1, img2, pair_label) ---
    # We need a way to get the *original* label of img1
    # This requires modifying the SiameseDataset or passing labels differently
    # Let's assume we pass a dataloader based on a simple dataset for visualization:

    print(f"Generating embeddings for visualization (Epoch {epoch_num})...")
    with torch.no_grad():
        for batch_data in dataloader:
            # Assuming dataloader yields (images, batch_labels)
            if len(batch_data) != 2:
                 print("Warning: Dataloader for visualize_embeddings did not yield (images, labels). Skipping batch.")
                 continue # Or adapt based on actual structure

            images, batch_labels = batch_data
            images = images.to(device).float() # Ensure correct type and device

            # Get embeddings from the base network
            outputs = model.base_network(images)
            embeddings.append(outputs.cpu()) # Move to CPU before appending
            true_labels.append(batch_labels.cpu())

    if not embeddings:
        print("No embeddings generated for visualization.")
        return

    embeddings = torch.cat(embeddings).numpy()
    true_labels = torch.cat(true_labels).numpy()

    print(f"Generated {embeddings.shape[0]} embeddings. Running t-SNE...")

    # Reduce dimensionality
    # Adjust perplexity based on number of samples - typical range 5-50
    perplexity_val = min(30, max(5, embeddings.shape[0] // 10)) # Heuristic
    tsne = TSNE(n_components=2, perplexity=perplexity_val, random_state=42, n_iter=300) # Add random_state
    reduced = tsne.fit_transform(embeddings)

    print("Plotting t-SNE...")
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=true_labels, cmap='tab10', alpha=0.7)

    # Create legend handles manually if needed (more robust)
    # unique_labels = sorted(list(set(true_labels)))
    # legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Class {l}',
    #                           markerfacecolor=plt.cm.tab10(l / max(unique_labels) if max(unique_labels) > 0 else 0), markersize=10)
    #                   for l in unique_labels]
    # plt.legend(handles=legend_handles, title="Classes")

    plt.legend(*scatter.legend_elements(), title="Classes") # Simpler legend
    plt.title(f't-SNE Visualization of Embeddings (Epoch {epoch_num})')
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    # Save the plot instead of just showing
    plot_filename = f'tsne_epoch_{epoch_num}.png'
    plt.savefig(plot_filename)
    print(f"Saved t-SNE plot to {plot_filename}")
    plt.close() # Close the figure to free memory

# --- Remember to update main.py to use this revised utils.py ---