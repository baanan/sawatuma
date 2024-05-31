from pathlib import Path
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

    path = Path("model.pickle")
    if path.exists():
        model = Model.load(path)
    else:
        train, test = sawatuma.datasets.listening_counts(
            parameters,
            train_fraction=0.7,
        )

        # this *massive* regularization term is necessary in order to prevent every guess going to 1
        model = Model(train.parameters, 64, 1, 512)
        model.train(train, test, num_epochs=10)
        model.save(path)

    print(model.similar_tracks(0)[:10])


if __name__ == "__main__":
    main()
