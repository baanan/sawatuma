import sawatuma.datasets
from sawatuma.model import Model


# USER_COUNT = 120_322
# TRACK_COUNT = 50_813_373
LISTENING_COUNTS_COUNT = 519_293_333


def main():
    # users = datasets.user_list()
    # tracks = datasets.track_list()

    parameters = sawatuma.datasets.Parameters(
        user_count=1000,
        track_count=10000,
        listening_counts_count=LISTENING_COUNTS_COUNT,
        dataset_divisor=2,
    )

    train, test = sawatuma.datasets.listening_counts(
        parameters,
        train_fraction=0.7,
    )

    model = Model(train, 64, 1, 512)

    model.train(10, test)

    # user, track, rating = test[0]
    # found = model(user, track)
    # print(f"found: {found}, expected: {rating}")

    # listen_count = sawatuma.datasets.ListenCount(3504, 16993728, 0, 0)
    # found = model(
    #     listen_count.user_one_hot(parameters).unsqueeze(0),
    #     listen_count.track_one_hot(parameters).unsqueeze(0),
    # )
    # print(f"found: {found}, expected: 0.0")


if __name__ == "__main__":
    main()
