"""Download and unzip the African Wildlife dataset into datasets/african-wildlife."""
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip"
ROOT = Path(__file__).resolve().parents[1] / "datasets"


def main():
    target = ROOT / "african-wildlife"
    if (target / "images" / "train").exists():
        print(f"Dataset already present at {target}")
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    zip_path = ROOT / "african-wildlife.zip"
    print(f"Downloading {URL} ...")
    urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        # the archive may or may not contain a top-level folder
        dest = ROOT if names[0].startswith("african-wildlife") else target
        zf.extractall(dest)
    zip_path.unlink()
    print(f"Dataset ready at {target}")


if __name__ == "__main__":
    main()
