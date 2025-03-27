from pathlib import Path
import shutil

def moveImagesToRoot(rootDir: str = "original_dataset") -> None:
    """Collect all images to root directory"""
    rootPath = Path(rootDir)
    for path in rootPath.rglob("*"):
        if path.is_file() and path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            destPath = rootPath / path.name
            if destPath.exists():
                base = destPath.stem
                ext = destPath.suffix
                i = 1
                while (rootPath / f"{base}_{i}{ext}").exists():
                    i += 1
                destPath = rootPath / f"{base}_{i}{ext}"
            shutil.move(str(path), str(destPath))

def removeSubdirs(rootDir: str = "original_dataset") -> None:
    """Remove all subdirectories"""
    rootPath = Path(rootDir)
    for dirPath in rootPath.glob("*"):
        if dirPath.is_dir() and dirPath != rootPath:
            shutil.rmtree(dirPath)

def createEmptyDirectory(dir: str) -> None:
    """Create empty directory, removing existing if needed"""
    path = Path(dir)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)

