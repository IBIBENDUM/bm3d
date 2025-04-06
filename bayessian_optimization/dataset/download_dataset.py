from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

from directory_funcs import moveImagesToRoot, removeSubdirs

def downloadDataset(datasetName: str="meriemelkhal/random",
                    downloadDir: str="original_dataset"):

    downloadPath = Path(downloadDir)
    
    if downloadPath.exists():
        print(f"Dataset already downloaded")
        return

    downloadPath.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(datasetName,
                               path=downloadDir,
                               unzip=True)

    moveImagesToRoot(downloadDir)
    removeSubdirs(downloadDir)

    print(f"Dataset {datasetName} downloaded to {downloadDir}")

if __name__ == "__main__":
    downloadDataset("prasunroy/natural-images")

