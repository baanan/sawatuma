import sawatuma.datasets
from sawatuma.model import Model


USER_COUNT = 120_322
TRACK_COUNT = 50_813_373
LISTENING_COUNTS_COUNT = 519_293_333


def main():
    # users = datasets.user_list()
    # tracks = datasets.track_list()

    parameters = sawatuma.datasets.Parameters(
        USER_COUNT,
        TRACK_COUNT,
        LISTENING_COUNTS_COUNT,
        rating_size=5,
        track_divisor=64,
        user_divisor=64,
    )

    train, test = sawatuma.datasets.listening_counts(
        parameters,
        train_fraction=0.7,
        transform=lambda counts: (
            counts.user_one_hot(parameters),
            counts.track_one_hot(parameters),
            counts.rating_one_hot(parameters),
        ),
    )

    model = Model(parameters, 16, 0.1, 0.05)

    for epoch in range(25):
        print(f"-- epoch {epoch + 1} --")
        model.train_once(train, batch_size=256)
        loss = model.evaluate(test)
        print(f"loss: {loss}")

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
