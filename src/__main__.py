from pathlib import Path
import sys
import sawatuma.datasets
from sawatuma.model import Model


# USER_COUNT = 120_322
# TRACK_COUNT = 50_813_373
LISTENING_COUNTS_COUNT = 519_293_333


def main():
    # users = datasets.user_list()
    # tracks = datasets.track_list()

    parameters = sawatuma.datasets.Parameters(
        user_count=10_000,
        track_count=10_000,
        listening_counts_count=LISTENING_COUNTS_COUNT,
        dataset_divisor=2,
    )

    path = Path("data/model.pickle")
    if len(sys.argv) > 1 and sys.argv[1] == "--refresh" or not path.exists():
        train, test = sawatuma.datasets.listening_counts(
            parameters,
            train_fraction=0.7,
        )

        # this *massive* regularization term is necessary in order to prevent every guess going to 1
        model = Model(train.parameters, 100, 5, 512)
        model.train(train, test, num_epochs=5)
        model.save(path)
    else:
        model = Model.load(path)

    track_info = sawatuma.datasets.TrackInfoDataset(download=True)

    similar_tracks = [track_info[track] for track in model.similar_tracks(7636)[:10]]
    first = similar_tracks[0]  # the most similar track will be itself
    print(f"similar tracks to {first.artist_name} - {first.track_name}:")
    for track in similar_tracks[1:]:
        print(f"  {track.artist_name} - {track.track_name}")


if __name__ == "__main__":
    main()
