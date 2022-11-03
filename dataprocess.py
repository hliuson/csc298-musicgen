import muspy
import torch

# https://muspy.readthedocs.io/en/latest/datasets/datasets.html
def get_dataset():
    data = muspy.datasets.MAESTRODatasetV3(root="data", download_and_extract=True)
    return data.to_pytorch_dataset(
        representation="pianoroll",
    )