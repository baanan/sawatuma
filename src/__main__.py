from torch.utils.data import DataLoader
import sawatuma.datasets
from sawatuma.device import device


USER_COUNT = 120_322
TRACK_COUNT = 50_813_373
LISTENING_COUNTS_COUNT = 519_293_333


def main():
    # users = datasets.user_list()
    # tracks = datasets.track_list()

    parameters = sawatuma.datasets.Parameters(
        USER_COUNT, TRACK_COUNT, LISTENING_COUNTS_COUNT, track_divisor=64
    )

    train, test = sawatuma.datasets.listening_counts(
        parameters,
        train_fraction=0.7,
        transform=lambda counts: (
            counts.user_one_hot(parameters),
            counts.track_one_hot(parameters),
            counts.rating(),
        ),
    )

    dataloader = DataLoader(train, batch_size=64, shuffle=True)

    print("getting the first item from the loader...")
    users, tracks, rating = next(iter(dataloader))
    print(users.size())
    print(tracks.size())


if __name__ == "__main__":
    main()
