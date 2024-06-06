import bz2
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import polars as pl
import sawatuma_rs

ROOT_DIR = "data"

USER_FILE = "users.tsv"
USER_MAPPING_FILE = "user_mapping.tsv"
USER_URL = "http://www.cp.jku.at/datasets/LFM-2b/chiir/users.tsv.bz2"

TRACK_FILE = "tracks.tsv"
TRACK_MAPPING_FILE = "track_mapping.tsv"
TRACK_URL = "http://www.cp.jku.at/datasets/LFM-2b/chiir/tracks.tsv.bz2"

LISTENING_COUNTS_FILE = "listening_counts.tsv"
LISTENING_COUNTS_FILE_FILTERED = "listening_counts_filtered"
LISTENING_COUNTS_FILE_TRAIN = "listening_counts_train.tsv"
LISTENING_COUNTS_FILE_TEST = "listening_counts_test.tsv"
LISTENING_COUNTS_URL = (
    "http://www.cp.jku.at/datasets/LFM-2b/chiir/listening-counts.tsv.bz2"
)


def _read_tsvbz2(url: str) -> pl.DataFrame:
    print(f"downloading file from url `{url}`...")
    path, _ = urllib.request.urlretrieve(url)
    print("  decompressing file...")
    with bz2.open(path) as decoded:
        decoded = decoded.read()
        print("  converting the file to a DataFrame...")
        frame = pl.read_csv(decoded, separator="\t", quote_char=None)
    os.remove(path)
    return frame


class NoFileFoundException(Exception):
    ...


def _get_or_download(
    file_name: str,
    url: str,
    *,
    root: str = ROOT_DIR,
    download: bool = True,
) -> pl.LazyFrame:
    os.makedirs(root, exist_ok=True)

    path = Path(root, file_name)
    if path.is_file():
        print(f"reading file `{path}`")
        return pl.scan_csv(path, separator="\t", quote_char=None)
    elif download:
        downloaded = _read_tsvbz2(url)
        downloaded.write_csv(path, separator="\t")
        return downloaded.lazy()
    else:
        raise NoFileFoundException


def user_list(*, root: str = ROOT_DIR, download: bool = True) -> pl.DataFrame:
    return _get_or_download(USER_FILE, USER_URL, root=root, download=download).collect()


@dataclass
class Parameters:
    user_count: int
    track_count: int
    listening_counts_count: int

    dataset_divisor: int


@dataclass
class ListenCount:
    user_id: int
    track_id: int
    count: int


class ListeningCountsDataset:
    def __init__(
        self,
        parameters: Parameters,
        *,
        train: bool,
        train_fraction: float,
        root: str = ROOT_DIR,
        download: bool = True,
    ):
        self.parameters = parameters

        if train:
            path = Path(root, LISTENING_COUNTS_FILE_TRAIN)
        else:
            path = Path(root, LISTENING_COUNTS_FILE_TEST)

        if path.is_file():
            print(f"reading file `{path}`")
            self.listening_counts = pl.read_csv(path)
            return

        print("listening counts haven't been separated, separating now")
        _get_or_download(
            LISTENING_COUNTS_FILE,
            LISTENING_COUNTS_URL,
            root=root,
            download=download,
        )

        print("cutting initial dataset in half (this may take a while!)")
        sawatuma_rs.cut_listening_counts(parameters.dataset_divisor, root=root)

        listening_counts = pl.scan_csv(
            f"{root}/{LISTENING_COUNTS_FILE_FILTERED}_{parameters.dataset_divisor}.tsv",
            separator="\t",
        )

        # a series of the most popular tracks in the dataset
        print(
            f"finding the {parameters.track_count} most popular tracks in the dataset"
        )
        top_tracks = (
            listening_counts.group_by("track_id")
            .agg(pl.col("user_id").count().alias("user_count"))
            .top_k(parameters.track_count, by="user_count")
            .select("track_id")
            .collect(streaming=True)
        )

        print("  writing the mapping to file")
        top_tracks.write_csv(f"{root}/{TRACK_MAPPING_FILE}", separator="\t")

        top_tracks = top_tracks["track_id"]

        print("  creating the output indicies of the tracks")
        track_indicies = pl.Series(values=list(range(parameters.track_count)))

        # rescan csv so the garbage collector doesn't have to keep around the old one
        listening_counts = pl.scan_csv(
            f"{root}/{LISTENING_COUNTS_FILE_FILTERED}_{parameters.dataset_divisor}.tsv",
            separator="\t",
        )

        # a series of the users with the most expansive track listens
        print(
            f"finding the {parameters.user_count} most listening users in the dataset"
        )
        top_users = (
            listening_counts.filter(pl.col("track_id").is_in(top_tracks))
            .group_by("user_id")
            .agg(pl.col("track_id").count().alias("track_count"))
            .top_k(parameters.user_count, by="track_count")
            .select("user_id")
            .collect(streaming=True)
        )

        print("  writing the mapping to file")
        top_users.write_csv(f"{root}/{USER_MAPPING_FILE}", separator="\t")

        top_users = top_users["user_id"]

        print("  creating the output indicies of the users")
        user_indicies = pl.Series(values=list(range(parameters.user_count)))

        # rescan csv so the garbage collector doesn't have to keep around the old one
        listening_counts = pl.scan_csv(
            f"{root}/{LISTENING_COUNTS_FILE_FILTERED}_{parameters.dataset_divisor}.tsv",
            separator="\t",
        )

        print("filtering the dataset by the maps")
        filtered_counts = (
            listening_counts.filter(
                (pl.col("track_id").is_in(top_tracks)).and_(
                    pl.col("user_id").is_in(top_users)
                )
            )
            .select(
                pl.col("track_id").replace(top_tracks, track_indicies),
                pl.col("user_id").replace(top_users, user_indicies),
                pl.col("count"),
            )
            .collect(streaming=True)
        )

        print("creating random numbers to split the dataset")
        random_list = np.random.rand(len(filtered_counts))
        print("  checking if they are over the fraction")
        is_train = pl.lit(random_list < train_fraction)

        print("filtering training counts")
        training_counts = filtered_counts.filter(is_train)
        print("  writing training counts")
        training_counts.write_csv(Path(root, LISTENING_COUNTS_FILE_TRAIN))

        print("filtering testing counts")
        testing_counts = filtered_counts.filter(is_train.not_())
        print("  writing testing counts")
        testing_counts.write_csv(Path(root, LISTENING_COUNTS_FILE_TEST))

        if train:
            self.listening_counts = training_counts
        else:
            self.listening_counts = testing_counts

    def get_listening_counts(self) -> pl.DataFrame:
        return self.listening_counts

    def __len__(self):
        return len(self.listening_counts)

    def __getitem__(self, index: int) -> ListenCount:
        track_id = self.listening_counts[index, 0]
        user_id = self.listening_counts[index, 1]
        count = self.listening_counts[index, 2]

        return ListenCount(user_id, track_id, count)


def listening_counts(
    parameters: Parameters,
    *,
    train_fraction: float,
    root: str = ROOT_DIR,
    download: bool = True,
) -> Tuple[ListeningCountsDataset, ListeningCountsDataset]:
    return ListeningCountsDataset(
        parameters,
        train=True,
        train_fraction=train_fraction,
        root=root,
        download=download,
    ), ListeningCountsDataset(
        parameters,
        train=False,
        train_fraction=train_fraction,
        root=root,
        download=download,
    )


class _ComponentDataset:
    def __init__(
        self,
        *,
        root: str = ROOT_DIR,
        download: bool = True,
    ):
        mapping_path = f"{root}/{self._mapping_file()}"
        # assume that the mapping file always exists when the list is made
        # the listening counts dataset outputs these info files when it's made
        self.map = pl.read_csv(mapping_path, separator="\t")

        if len(self.map.get_columns()) <= 1:
            print("  info hasn't been merged, merging it now")
            # the info dataset hasn't been merged yet
            info = self._get_info(root, download)
            # "left" ensures that only the tracks in the map are preserved
            self.map = self.map.join(info, on="track_id", how="left")
            # save it for later
            self.map.write_csv(mapping_path, separator="\t")

    @staticmethod
    def _mapping_file() -> str:
        ...

    @staticmethod
    def _get_info(root: str, download: bool) -> pl.DataFrame:
        ...

    def __len__(self) -> int:
        return len(self.map)

    def __getitem__(self, index: int) -> Any:
        ...


@dataclass
class Track:
    track_id: int
    original_track_id: int
    artist_name: str
    track_name: str

    def __repr__(self) -> str:
        return f"{self.artist_name} - {self.track_name} (#{self.track_id})"


class TrackInfo(_ComponentDataset):
    def __init__(
        self,
        *,
        root: str = ROOT_DIR,
        download: bool = True,
    ):
        print("reading track info")
        super().__init__(root=root, download=download)

    @staticmethod
    def _mapping_file() -> str:
        return "track_mapping.tsv"

    @staticmethod
    def _get_info(root: str, download: bool) -> pl.DataFrame:
        return _get_or_download(
            TRACK_FILE, TRACK_URL, root=root, download=download
        ).collect()

    def search(self, components: List[str]) -> List[Track]:
        filtered = self.map.with_row_index().filter(
            (pl.col("artist_name").str.contains_any(components)).or_(
                pl.col("track_name").str.contains_any(components)
            )
        )
        return [
            Track(index, track_id, artist_name, track_name)
            for index, track_id, artist_name, track_name in filtered.rows()
        ]

    def __getitem__(self, index: int) -> Track:
        track_id = self.map[index, 0]
        artist_name = self.map[index, 1]
        track_name = self.map[index, 2]
        return Track(index, track_id, artist_name, track_name)
