"""Downloads the UC Merced Land Use dataset and splits it into training and validation sets.

Usage:
    uv run scripts/prepare_data.py --val-size 0.2
"""
import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from sklearn.model_selection import train_test_split

DATASET_URL = "https://huggingface.co/datasets/torchgeo/ucmerced/resolve/main/UCMerced_LandUse.zip"
REPO_ROOT = Path(__file__).resolve().parents[1]


def download_dataset(data_path: Path) -> Path:
    """Downloads and extracts the dataset, keeping one folder per class under data_path/raw.

    Skips the download if data_path/raw already exists.

    Args:
        data_path: Directory that holds all dataset folders.

    Returns:
        Path to the folder with one subfolder per class.
    """
    raw_path = data_path / "raw"
    if raw_path.exists():
        print(f"Found existing dataset, skipping download: raw_path={raw_path}")
        return raw_path

    data_path.mkdir(parents=True, exist_ok=True)
    zip_path = data_path / "UCMerced_LandUse.zip"
    print(f"Downloading dataset: url={DATASET_URL}")
    urllib.request.urlretrieve(DATASET_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zip_file:
        zip_file.extractall(data_path)
    (data_path / "UCMerced_LandUse" / "Images").rename(raw_path)
    shutil.rmtree(data_path / "UCMerced_LandUse")
    zip_path.unlink()

    return raw_path


def copy_images(paths: list[Path], destination_dir: Path):
    """Copies images into per-class subfolders of a destination directory.

    Each image is placed under destination_dir/<class_name>/, where class_name
    is the name of the image's parent folder. This matches the layout expected
    by torchvision.datasets.ImageFolder.

    Args:
        paths: List of image paths to copy.
        destination_dir: Directory where images will be copied.
    """
    for path in paths:
        class_dir = destination_dir / path.parent.name
        class_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, class_dir)


def split_dataset(raw_path: Path, data_path: Path, val_size: float, seed: int):
    """Splits the images into data_path/train and data_path/val, keeping class ratios equal.

    Args:
        raw_path: Folder with one subfolder of images per class.
        data_path: Directory where the train and val folders are created.
        val_size: Fraction of the images used for validation.
        seed: Seed for the random split.
    """
    image_paths = sorted(raw_path.glob("*/*.tif"))
    # Parent folder name is the class label, used to keep class ratios equal across splits
    labels = [path.parent.name for path in image_paths]

    train_paths, val_paths = train_test_split(
        image_paths, test_size=val_size, random_state=seed, stratify=labels
    )

    for split_name, split_paths in [("train", train_paths), ("val", val_paths)]:
        split_path = data_path / split_name
        # A previous split with a different seed or source would leave images in both sets
        if split_path.exists():
            shutil.rmtree(split_path)
        copy_images(split_paths, split_path)
        print(f"Copied images: split={split_name} count={len(split_paths)} path={split_path}")


def main():
    parser = argparse.ArgumentParser(description="Download and split the UC Merced Land Use dataset.")
    parser.add_argument("--data-path", type=Path, default=REPO_ROOT / "data", help="Directory for all dataset folders.")
    parser.add_argument("--val-size", type=float, default=0.2, help="Fraction of the images used for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random split.")
    args = parser.parse_args()

    raw_path = download_dataset(args.data_path)
    split_dataset(raw_path, args.data_path, args.val_size, args.seed)


if __name__ == "__main__":
    main()
