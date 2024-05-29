import polars as pl

NEW_USER_COUNT = 2000

print("reading listening counts")
training_counts = pl.read_csv("data/listening_counts_train.tsv")

print("finding all tracks where which more than 2000 users listen to")
user_counts = (
    training_counts.lazy()
    .group_by("track_id")
    .agg(pl.col("user_id").count().alias("user_count"))
    .filter(pl.col("user_count").gt(NEW_USER_COUNT))
    .collect()
)

print("  converting those tracks to a dictionary")
in_user_counts = pl.col("track_id").is_in(user_counts["track_id"])

print("filtering training counts by the dictionary")
training_counts = training_counts.filter(in_user_counts)
print("  writing training counts")
training_counts.write_csv("data/listening_counts_train.tsv")

print("reading testing counts")
testing_counts = pl.read_csv("data/listening_counts_test.tsv")
print("  filtering testing counts")
testing_counts = testing_counts.filter(in_user_counts)
print("  writing testing counts")
testing_counts.write_csv("data/listening_counts_test.tsv")
