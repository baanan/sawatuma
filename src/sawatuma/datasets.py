import bz2
from math import floor
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import polars as pl
import torch
from sawatuma.device import device
from torch import Tensor
from torch.utils.data import Dataset

ROOT_DIR = "data"

USER_FILE = "users.tsv"
USER_URL = "http://www.cp.jku.at/datasets/LFM-2b/chiir/users.tsv.bz2"

TRACK_FILE = "tracks.tsv"
TRACK_URL = "http://www.cp.jku.at/datasets/LFM-2b/chiir/tracks.tsv.bz2"

LISTENING_COUNTS_FILE = "listening_counts.tsv"
LISTENING_COUNTS_FILE_TRAIN = "listening_counts_train.tsv"
LISTENING_COUNTS_FILE_TEST = "listening_counts_test.tsv"
LISTENING_COUNTS_URL = (
    "http://www.cp.jku.at/datasets/LFM-2b/chiir/listening-counts.tsv.bz2"
)


def __read_tsvbz2__(url: str) -> pl.DataFrame:
    print(f"downloading file from url `{url}`...")
    path, _ = urllib.request.urlretrieve(url)
    print("  decompressing file...")
    with bz2.open(path) as decoded:
        decoded = decoded.read()
        print("  converting the file to a DataFrame...")
        frame = pl.read_csv(decoded, separator="\t")
    os.remove(path)
    return frame


class NoFileFoundException(Exception):
    ...


def __get_or_download__(
    file_name: str, url: str, *, root: str = ROOT_DIR, download: bool = True
) -> pl.LazyFrame:
    os.makedirs(root, exist_ok=True)

    path = Path(root, file_name)
    if path.is_file():
        print(f"reading file `{path}`")
        return pl.scan_csv(path, separator="\t")
    elif download:
        downloaded = __read_tsvbz2__(url)
        downloaded.write_csv(path, separator="\t")
        return downloaded.lazy()
    else:
        raise NoFileFoundException


def user_list(*, root: str = ROOT_DIR, download: bool = True) -> pl.DataFrame:
    return __get_or_download__(
        USER_FILE, USER_URL, root=root, download=download
    ).collect()


def track_list(*, root: str = ROOT_DIR, download: bool = True) -> pl.DataFrame:
    return __get_or_download__(
        TRACK_FILE, TRACK_URL, root=root, download=download
    ).collect()


@dataclass
class Parameters:
    user_count: int
    total_track_count: int
    listening_counts_count: int
    track_divisor: int = 1

    def track_count(self) -> int:
        return floor(self.total_track_count / self.track_divisor)

    def track_is_valid(self, index: int) -> bool:
        return index % self.track_divisor == 0 and index <= self.total_track_count

    def track_dataset_to_internal(self, index: int) -> int:
        return floor(index / self.track_divisor)

    def track_internal_to_dataset(self, index: int) -> int:
        return index * self.track_divisor


def __make_one_hot__(length: int, index: int) -> torch.Tensor:
    return torch.zeros([length], device=device).scatter(
        0, torch.tensor(index, device=device), 1
    )


@dataclass
class ListenCount:
    user_id: int
    raw_track_id: int
    count: int

    def track_id(self, parameters: Parameters) -> int:
        return parameters.track_dataset_to_internal(self.raw_track_id)

    def rating(self) -> int:
        return 1 if self.count >= 2 else 0

    def user_one_hot(self, parameters: Parameters) -> Tensor:
        return __make_one_hot__(parameters.user_count, self.user_id)

    def track_one_hot(self, parameters: Parameters) -> Tensor:
        return __make_one_hot__(parameters.track_count(), self.track_id(parameters))


class ListeningCountsDataset(Dataset):
    def __init__(
        self,
        parameters: Parameters,
        *,
        train: bool,
        train_fraction: float,
        root: str = ROOT_DIR,
        download: bool = True,
        transform: Optional[Callable[[ListenCount], Any]] = None,
    ):
        self.transform = transform

        if train:
            path = Path(root, LISTENING_COUNTS_FILE_TRAIN)
        else:
            path = Path(root, LISTENING_COUNTS_FILE_TEST)

        if path.is_file():
            print(f"reading file `{path}`")
            self.listening_counts = pl.read_csv(path)
            return

        print("listening counts haven't been separated, separating now")
        listening_counts = __get_or_download__(
            LISTENING_COUNTS_FILE,
            LISTENING_COUNTS_URL,
            root=root,
            download=download,
        )

        print("creating random numbers...")
        random_list = np.random.rand(parameters.listening_counts_count)
        print("checking if they are over the fraction...")
        is_train = pl.lit(random_list < train_fraction)

        valid_index = pl.col("track_id").mod(parameters.track_divisor).eq(0)

        training_counts = listening_counts.filter(valid_index.and_(is_train))
        testing_counts = listening_counts.filter(valid_index.and_(is_train.not_()))

        print("collecting training counts...")
        training_counts = training_counts.collect(streaming=True)
        print("  writing them to disk...")
        training_counts.write_csv(Path(root, LISTENING_COUNTS_FILE_TRAIN))
        print("collecting testing counts...")
        testing_counts = testing_counts.collect(streaming=True)
        print("  writing them to disk...")
        testing_counts.write_csv(Path(root, LISTENING_COUNTS_FILE_TEST))

        if train:
            self.listening_counts = training_counts
        else:
            self.listening_counts = testing_counts

    def __len__(self):
        return len(self.listening_counts)

    def __getitem__(self, index) -> Any:
        counts = ListenCount(
            self.listening_counts[index, 0],
            self.listening_counts[index, 1],
            self.listening_counts[index, 2],
        )
        if self.transform is not None:
            counts = self.transform(counts)
        return counts


def listening_counts(
    parameters: Parameters,
    *,
    train_fraction: float,
    root: str = ROOT_DIR,
    download: bool = True,
    transform: Optional[Callable[[ListenCount], Any]] = None,
) -> Tuple[ListeningCountsDataset, ListeningCountsDataset]:
    return ListeningCountsDataset(
        parameters,
        train=True,
        train_fraction=train_fraction,
        root=root,
        download=download,
        transform=transform,
    ), ListeningCountsDataset(
        parameters,
        train=False,
        train_fraction=train_fraction,
        root=root,
        download=download,
        transform=transform,
    )
