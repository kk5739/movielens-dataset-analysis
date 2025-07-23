import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window

def main(in_path, out_path, train_r, val_r, test_r, seed, min_ratings):
    spark = (
        SparkSession.builder
        .appName("SplitRatingsPerUser")
        .getOrCreate()
    )

    # Load ratings
    ratings = spark.read.parquet(in_path)

    # Filter out tiny users from training (but keep them for test)
    active_users = ratings.groupBy("userId").count().filter(F.col("count") >= min_ratings).select("userId")
    ratings_filtered = ratings.join(active_users, on="userId", how="inner")

    # Add a random order within each active user
    w = Window.partitionBy("userId").orderBy("rand")
    ratings_filtered = ratings_filtered.withColumn("rand", F.rand(seed))

    # Row number & per-user count
    w_all = Window.partitionBy("userId")
    ratings_filtered = ratings_filtered.withColumn("rn", F.row_number().over(w)) \
                                      .withColumn("cnt", F.count("*").over(w_all))

    # Compute fraction index in [0,1)
    ratings_filtered = ratings_filtered.withColumn("frac", (F.col("rn") - 1) / F.col("cnt"))

    # Assign split for active users
    ratings_filtered = ratings_filtered.withColumn(
        "split",
        F.when(F.col("frac") < train_r, "train")
         .when(F.col("frac") < train_r + val_r, "val")
         .otherwise("test")
    )

    # Cache for efficient writing
    ratings_filtered.cache()

    # **Print split counts (before dropping columns)**
    # ratings per split (rows)
    ratings_filtered.groupBy("split").count().show()

    # distinct users per split
    ratings_filtered.groupBy("split").agg(F.countDistinct("userId").alias("user_cnt")).show()

    # Write active users
    (ratings_filtered.filter("split = 'train'")
                    .drop("rand","rn","cnt","frac","split")
                    .repartition(200)
                    .write.mode("overwrite")
                    .parquet(f"{out_path}/train"))

    (ratings_filtered.filter("split = 'val'")
                    .drop("rand","rn","cnt","frac","split")
                    .write.mode("overwrite")
                    .parquet(f"{out_path}/val"))

    (ratings_filtered.filter("split = 'test'")
                    .drop("rand","rn","cnt","frac","split")
                    .write.mode("overwrite")
                    .parquet(f"{out_path}/test"))

    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Usage: spark-submit split.py <input_parquet> <output_dir> "
                 "[train_ratio val_ratio test_ratio seed min_ratings]")

    in_path   = sys.argv[1]
    out_path  = sys.argv[2]
    train_r   = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
    val_r     = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
    test_r    = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1
    seed      = int(sys.argv[6])   if len(sys.argv) > 6 else 42
    min_ratings = int(sys.argv[7]) if len(sys.argv) > 7 else 500

    assert abs(train_r + val_r + test_r - 1.0) < 1e-6, "Splits must sum to 1"
    main(in_path, out_path, train_r, val_r, test_r, seed, min_ratings)
