from torch.utils.data import DataLoader
import sawatuma.datasets
from sawatuma.device import device


USER_COUNT = 120_322
TRACK_COUNT = 50_813_373


def main():
    # users = datasets.user_list()
    # tracks = datasets.track_list()

    parameters = sawatuma.datasets.Parameters(USER_COUNT, TRACK_COUNT)

    print(device)

    # train, test = sawatuma.datasets.listening_counts(
    #     train_fraction=0.7,
    #     transform=lambda counts: (
    #         counts.user_one_hot(parameters),
    #         counts.track_one_hot(parameters),
    #         counts.rating(),
    #     ),
    # )

    print(sawatuma.datasets.ListenCount(1, 1, 1).user_one_hot(parameters))

    # # with a batch size of 16, the track one hots themselves will take around 1gb
    # dataloader = DataLoader(train, batch_size=2, shuffle=True)
    #
    # print("getting the first item from the loader...")
    # users, tracks, rating = next(iter(dataloader))
    # print(users.size())


if __name__ == "__main__":
    main()
