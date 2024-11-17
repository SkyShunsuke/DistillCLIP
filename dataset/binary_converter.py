import os
import csv
from torchvision.datasets import STL10
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from stl import LABEL_TO_CLASSNAME, STL10Dataset

def create_output_folders(output_dir, label_names):
    """
    Creates output directories for labeled data.
    """
    for label in label_names:
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)


def save_images_and_labels(dataset, output_dir, labels_file, labeled=True):
    """
    Converts and saves images from the STL10 dataset to JPEG format.
    Saves corresponding labels in a CSV file.
    
    Args:
        dataset: STL10 dataset instance.
        output_dir: Output directory to save the images.
        labels_file: Path to the labels CSV file.
        labeled: Boolean, whether the dataset is labeled.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(labels_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "path", "label"])

        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            image = data["image"]
            label = data["label"].item()
            if labeled:
                class_name = LABEL_TO_CLASSNAME[label]
                folder_path = os.path.join(output_dir, class_name)
                os.makedirs(folder_path, exist_ok=True)
                file_path = os.path.join(folder_path, f"{i}.jpeg")
            else:
                # For unlabeled data
                file_path = os.path.join(output_dir, f"{i}.jpeg")
                label = -1  # Unlabeled

            save_image(image, file_path)
            writer.writerow([i, file_path, label])


def main():
    transform = transforms.Compose([
        transforms.Resize((96, 96)),  # STL10 images are 96x96
        transforms.ToTensor(),
    ])

    root_dir = "data"
    output_dir = "stl10_images"

    # Prepare STL10 datasets
    train_dataset = STL10Dataset(root=root_dir, transform=transform, train=True, download=True)
    test_dataset = STL10Dataset(root=root_dir, transform=transform, train=False, download=True)
    unlabeled_dataset = STL10Dataset(root=root_dir, transform=transform, train=True, with_unlabeled=True, download=False)
    
    # Convert labeled data
    labeled_output_dir = os.path.join(output_dir, "train")
    test_labeled_output_dir = os.path.join(output_dir, "test")
    labeled_labels_file = os.path.join(output_dir, "labels_train.csv")
    print("Converting labeled data...")
    save_images_and_labels(train_dataset, labeled_output_dir, labeled_labels_file, labeled=True)
    
    # Convert test data
    print("Converting test data...")
    test_labels_file = os.path.join(output_dir, "labels_test.csv")
    save_images_and_labels(test_dataset, test_labeled_output_dir, test_labels_file, labeled=True)

    # Convert unlabeled data
    unlabeled_output_dir = os.path.join(output_dir, "unlabeled")
    test_unlabeled_output_dir = os.path.join(output_dir, "test_unlabeled")
    unlabeled_labels_file = os.path.join(output_dir, "labels_unlabeled.csv")
    
    print("Converting unlabeled data...")
    save_images_and_labels(unlabeled_dataset, unlabeled_output_dir, unlabeled_labels_file, labeled=False)

    print("Conversion complete!")


if __name__ == "__main__":
    main()
