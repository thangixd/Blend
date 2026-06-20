import argparse
import os
import tarfile
import urllib.request

from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def post_process_dataset(dataset_path: str):
    for table in os.listdir(dataset_path):
        os.rename(
            f"{dataset_path}/{table}", f"{dataset_path}/{table.split('_SEP_')[1]}"
        )
    print("Processed the dataset")


def extract_tar(tar_name: str, extract_path="."):
    try:
        with tarfile.open(tar_name, "r") as tar:
            tar.extractall(path=extract_path)
            print(f"Extracted all files to '{extract_path}'")
        os.remove(tar_name)
        print(f"Removed the tar file: '{tar_name}'")
    except Exception as e:
        print(f"An error occurred: {e}")


dataset_tar_names = {
    "public_bi": "pneuma_public_bi.tar",
    "chicago": "pneuma_chicago_10K.tar",
    "chembl": "pneuma_chembl_10K.tar",
    "fetaqa": "pneuma_fetaqa.tar",
    "adventure": "pneuma_adventure_works.tar",
    "bird": "pneuma_bird.tar",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This program downloads datasets used in the experiments.",
    )
    parser.add_argument("-d", "--dataset", default="all")
    dataset = parser.parse_args().dataset

    if dataset == "all":
        for _, tar_name in tqdm(dataset_tar_names.items()):
            urllib.request.urlretrieve(
                f"https://storage.googleapis.com/pneuma_open/{tar_name}",
                filename=os.path.join(SCRIPT_DIR, tar_name),
            )
            extract_tar(os.path.join(SCRIPT_DIR, tar_name), SCRIPT_DIR)
            post_process_dataset(os.path.join(SCRIPT_DIR, tar_name[:-4]))
    else:
        try:
            tar_name = dataset_tar_names[dataset]
            urllib.request.urlretrieve(
                f"https://storage.googleapis.com/pneuma_open/{tar_name}",
                filename=os.path.join(SCRIPT_DIR, tar_name),
            )
            extract_tar(os.path.join(SCRIPT_DIR, tar_name), SCRIPT_DIR)
            post_process_dataset(os.path.join(SCRIPT_DIR, tar_name[:-4]))
        except KeyError:
            print(
                f"Dataset {dataset} not found!"
            )
