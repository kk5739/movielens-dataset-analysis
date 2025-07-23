import sys
from typing import List

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics

def init_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("ALS‑MovieLens")
        .getOrCreate()
    )


def load_splits(spark: SparkSession, base_path: str):
    train = spark.read.parquet(f"{base_path}/train")
    val   = spark.read.parquet(f"{base_path}/val")
    test  = spark.read.parquet(f"{base_path}/test")
    return train.cache(), val.cache(), test.cache()

def train_best_als(train_df, val_df, ranks: List[int], regs: List[float]):
    """Grid‑search ALS hyper‑parameters on validation RMSE."""
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    best = {"rmse": float("inf")}

    for r in ranks:
        for reg in regs:
            als = ALS(rank=r,
                      regParam=reg,
                      maxIter=20,
                      userCol="userId",
                      itemCol="movieId",
                      ratingCol="rating",
                      coldStartStrategy="drop",
                      seed=42)
            model = als.fit(train_df)
            rmse = evaluator.evaluate(model.transform(val_df))
            print(f"ALS rank={r:3d}, λ={reg:<6} → val RMSE {rmse:.4f}")
            if rmse < best["rmse"]:
                best.update(rank=r, reg=reg, rmse=rmse, model=model)

    print(f"\n▶ best ALS: rank={best['rank']}, λ={best['reg']} (val RMSE {best['rmse']:.4f})\n")
    return best["model"]


def ranking_metrics_als(model, test_df, top_n: int):
    """Compute Precision / MAP / NDCG@k for ALS recommendations."""
    user_recs = model.recommendForAllUsers(top_n)
    preds = user_recs.select(
        "userId",
        F.expr("transform(recommendations, x -> x.movieId)").alias("pred")
    )

    truth = (
        test_df.groupBy("userId")
                .agg(F.collect_set("movieId").alias("truth"))
                .filter(F.size("truth") > 0)
    )

    pred_truth_rdd = (
        preds.join(truth, "userId")
             .select("pred", "truth")
             .rdd.map(lambda row: (row["pred"], row["truth"]))
    )
    metrics = RankingMetrics(pred_truth_rdd)
    print(f"ALS ranking (top‑{top_n}):")
    print(f"  Precision@{top_n:<3}: {metrics.precisionAt(top_n):.6f}")
    print(f"  MAP@{top_n:<3}:       {metrics.meanAveragePrecisionAt(top_n):.6f}")
    print(f"  NDCG@{top_n:<3}:      {metrics.ndcgAt(top_n):.6f}\n")


def regression_metrics(model, test_df):
    rmse = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")\
              .evaluate(model.transform(test_df))
    print(f"ALS test RMSE: {rmse:.6f}\n")

def main(splits_dir: str, top_n: int, ranks: List[int], regs: List[float]):
    spark = init_spark()
    print("#"*60, "\nSpark session started\n", "#"*60)

    train_df, val_df, test_df = load_splits(spark, splits_dir)
    print("Splits loaded & cached\n")

    best_model = train_best_als(train_df, val_df, ranks, regs)

    print("#"*60, "\nEvaluating best model on test set\n", "#"*60)
    regression_metrics(best_model, test_df)
    ranking_metrics_als(best_model, test_df, top_n)

    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: spark-submit als_model.py <splits_dir> "
                 "[top_n=100] [ranks=50,100] [regs=0.05,0.1]")

    splits_dir = sys.argv[1]
    top_n      = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    ranks       = list(map(int, sys.argv[3].split(","))) if len(sys.argv) > 3 else [50,100,160,200]
    regs        = list(map(float, sys.argv[4].split(","))) if len(sys.argv) > 4 else [0.05,0.1]

    main(splits_dir, top_n, ranks, regs)