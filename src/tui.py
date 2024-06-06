from typing import Iterator, Optional, TypeVar

from sawatuma.datasets import TrackInfo
from sawatuma.model import Model
import questionary


def run(track_info: TrackInfo, model: Model):
    while True:
        line = questionary.text("", qmark=">").ask()
        command = split(line)
        if not handle(command, track_info, model):
            break


def split(line: str) -> Iterator[str]:
    return iter(line.split())


T = TypeVar("T")


def maybe_next(iter: Iterator[T]) -> Optional[T]:
    try:
        return next(iter)
    except StopIteration:
        return None


def handle(input: Iterator[str], track_info: TrackInfo, model: Model) -> bool:
    command = maybe_next(input)

    match command:
        case "help":
            print(
                """Commands:
- exit
- search {term}"""
            )
        case "exit":
            return False
        case "search":
            search = track_info.search(list(input))
            choices = {str(track): track for track in search}
            selected = questionary.select("Which track?", choices, qmark="> ").ask()  # type: ignore
            if selected is not None:
                selected = choices[selected]
                print_similar(selected.track_id, track_info, model)
        case None:
            ...
        case _:
            print(f"unrecognized command: {command}")

    return True


def print_similar(id: int, track_info: TrackInfo, model: Model):
    similar_tracks = [track_info[track] for track in model.similar_tracks(id)[:10]]
    first = similar_tracks[0]  # the most similar track will be itself
    print(f"Similar tracks to {first.artist_name} - {first.track_name}:")
    for index, track in enumerate(similar_tracks[1:]):
        print(f" {index + 1}. {track.artist_name} - {track.track_name}")
