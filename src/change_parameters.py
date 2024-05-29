import polars as pl
import sawatuma.datasets

USER_COUNT = 120_322
TRACK_COUNT = 50_813_373
LISTENING_COUNTS_COUNT = 519_293_333

NEW_PARAMETERS = sawatuma.datasets.Parameters(
    USER_COUNT,
    TRACK_COUNT,
    LISTENING_COUNTS_COUNT,
    track_divisor=64,
    user_divisor=16,
)

user_in_bounds = pl.col("user_id").mod(NEW_PARAMETERS.user_divisor).eq(0)
track_in_bounds = pl.col("track_id").mod(NEW_PARAMETERS.track_divisor).eq(0)
in_bounds = user_in_bounds.and_(track_in_bounds)

training_counts = pl.read_csv("data/listening_counts_train.tsv")
training_counts = training_counts.filter(in_bounds)
training_counts.write_csv(("data/listening_counts_train.tsv"))

testing_counts = pl.read_csv("data/listening_counts_test.tsv")
testing_counts = testing_counts.filter(in_bounds)
testing_counts.write_csv(("data/listening_counts_test.tsv"))
