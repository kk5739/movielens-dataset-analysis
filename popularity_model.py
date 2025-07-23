import sys
from pyspark.sql import SparkSession, functions as F
from pyspark.mllib.evaluation import RankingMetrics

def init_spark():
    return (
        SparkSession.builder
        .appName("PopularityBaseline large‑scale")
        .getOrCreate()
    )

def load_split_dfs(spark, base_path):
    train_df = spark.read.parquet(f"{base_path}/train")
    val_df   = spark.read.parquet(f"{base_path}/val")
    test_df  = spark.read.parquet(f"{base_path}/test")
    return train_df, val_df, test_df

def get_top_n_movies(train_df, n):
    movie_counts = (
        train_df.groupBy("movieId")
        .count()
        .orderBy(F.desc("count"))
        .limit(n)
    )
    return [row["movieId"] for row in movie_counts.collect()]

def prediction_label_rdd(df, top_n_bc):
    user_labels = (
        df.groupBy("userId")
        .agg(F.collect_set("movieId").alias("label_movies"))
        .filter(F.size("label_movies") > 0)   # keep same filter logic as ALS
    )
    return (
        user_labels.rdd
        .map(lambda row: (top_n_bc.value, row["label_movies"]))
    )

def evaluate(df, split_name, top_n_bc):
    rdd = prediction_label_rdd(df, top_n_bc)
    metrics = RankingMetrics(rdd)
    print(f" {split_name} set:")
    print(f"  Precision@{len(top_n_bc.value):<3}: "
          f"{metrics.precisionAt(len(top_n_bc.value)):.4f}")
    print(f"  MAP@{len(top_n_bc.value):<3}:       "
          f"{metrics.meanAveragePrecisionAt(len(top_n_bc.value)):.4f}")
    print(f"  NDCG@{len(top_n_bc.value):<3}:      "
          f"{metrics.ndcgAt(len(top_n_bc.value)):.4f}\n")

def main(base_path, n_top, seed):
    spark = init_spark()
    print("#"*50)
    print("Spark started")
    print("#"*50)

    train_df, val_df, test_df = load_split_dfs(spark, base_path)
    print("#"*50)
    print("Splits loaded")
    print("#"*50)

    train_df.cache(); val_df.cache(); test_df.cache()

    top_n_list = get_top_n_movies(train_df, n_top)
    top_n_bc   = spark.sparkContext.broadcast(top_n_list)
    print(f"Top‑{n_top} list ready ({len(top_n_list)} items)\n")

    evaluate(val_df,   "Validation", top_n_bc)
    evaluate(test_df,  "Test",       top_n_bc)

    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: spark-submit popularity_model.py <splits_dir> "
                 "[top_n=100] [seed=42]")

    base_path = sys.argv[1]
    n_top     = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    seed      = int(sys.argv[3]) if len(sys.argv) > 3 else 42

    main(base_path, n_top, seed)
