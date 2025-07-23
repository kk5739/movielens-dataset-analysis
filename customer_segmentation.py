import sys, os
import random

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_set, count, size, array_intersect, corr, array_union
from pyspark.ml.feature import MinHashLSH
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf

rating_theshold = 2500

def to_sparse_factory(dim):
    def to_sparse(movies):
        idx = sorted(movies)
        vals = [1.0] * len(idx)
        return Vectors.sparse(dim, idx, vals)
    return udf(to_sparse, VectorUDT())

def question_1_spark(ratings_path, output_folder="results"):
    # Initialize Spark and load data
    spark = SparkSession.builder.appName("MovieTwinsQ1").getOrCreate()
    
    # Load and cast ratings data
    ratings = spark.read.parquet(ratings_path).select(
        col("userId").cast("int"),
        col("movieId").cast("int"),
        col("rating").cast("double")
    )
    
    # Filter for active users (with at least MIN_RATINGS)
    active_users = ratings.groupBy("userId").count().filter(col("count") >= rating_theshold).select("userId")
    ratings_filtered = ratings.join(active_users, on="userId", how="inner")
    
    # Group movies by user
    user_movies = ratings_filtered.groupBy("userId").agg(collect_set("movieId").alias("movies"))
    
    # Create sparse vectors for LSH
    max_movie_id = ratings.agg({"movieId": "max"}).collect()[0][0] + 1
    to_sparse = to_sparse_factory(max_movie_id)
    data = user_movies.withColumn("features", to_sparse("movies"))
    
    # Apply MinHashLSH to find similar users
    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
    model = mh.fit(data)
    
    # Find similar pairs (where u < v to avoid duplicates)
    similar_pairs = model.approxSimilarityJoin(data, data, 1, distCol="jaccardDist") \
        .filter(col("datasetA.userId") < col("datasetB.userId")) \
        .select(
            col("datasetA.userId").alias("user_1"),
            col("datasetB.userId").alias("user_2"),
            col("datasetA.movies").alias("movies_1"),
            col("datasetB.movies").alias("movies_2"),
            col("jaccardDist")
        )
    
    # Compute similarity metrics and get top 100 pairs
    top_pairs = similar_pairs \
        .withColumn("jaccardSim", 1.0 - col("jaccardDist")) \
        .withColumn("commonCount", size(array_intersect(col("movies_1"), col("movies_2")))) \
        .orderBy(col("jaccardSim").desc(), col("commonCount").desc()) \
        .select("user_1", "user_2", "commonCount", "jaccardSim") \
        .limit(100)
    
    # Save results and stop Spark session
    top_pairs.write.csv(os.path.join(output_folder, "q1.csv"), header=True, mode="overwrite")
    spark.stop()

#############    
# Question 2 - Correlation
#############

def question_2_spark(ratings_path, output_folder="results"):
    spark = SparkSession.builder.appName("MovieTwinsQ2").getOrCreate()
    ratings = spark.read.parquet(ratings_path).select(
        col("userId").cast("int"),
        col("movieId").cast("int"),
        col("rating").cast("double")
    )

    top100 = spark.read.csv(os.path.join(output_folder, "q1.csv"), header=True, inferSchema=True)
    ratingsA = ratings.select(col("userId").alias("user_1"), col("movieId"), col("rating").alias("rating_1"))
    ratingsB = ratings.select(col("userId").alias("user_2"), col("movieId"), col("rating").alias("rating_2"))

    joined_top = top100.select(col("user_1").alias("user_1"), col("user_2").alias("user_2")) \
        .join(ratingsA, on="user_1") \
        .join(ratingsB, on=["user_2", "movieId"])
    corr_top = joined_top.groupBy("user_1", "user_2").agg(corr("rating_1", "rating_2").alias("pearson"))
    avg_top = corr_top.agg({"pearson": "avg"}).collect()[0][0]

    user_ids = [row.userId for row in ratings.select("userId").distinct().collect()]
    rand_pairs = set()
    while len(rand_pairs) < 100:
        a, b = random.sample(user_ids, 2)
        rand_pairs.add(tuple(sorted((a, b))))
    rand_df = spark.createDataFrame(list(rand_pairs), ["user_1", "user_2"])

    joined_rand = rand_df.join(ratingsA, on="user_1").join(ratingsB, on=["user_2", "movieId"])
    corr_rand = joined_rand.groupBy("user_1", "user_2").agg(corr("rating_1", "rating_2").alias("pearson"))
    avg_rand = corr_rand.agg({"pearson": "avg"}).collect()[0][0]

    print(f"Average correlation for top 100 pairs: {avg_top:.4f}")
    print(f"Average correlation for random 100 pairs: {avg_rand:.4f}")
    
    # Create a DataFrame to save the correlation results
    correlation_results = spark.createDataFrame([
        ("top_100_pairs", float(avg_top)),
        ("random_100_pairs", float(avg_rand))
    ], ["pair_type", "avg_correlation"])

    # Save the correlation results to CSV
    correlation_results.write.csv(os.path.join(output_folder, "correlation_comparison.csv"), 
                                header=True, 
                                mode="overwrite")
    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: spark-submit customer_segmentation.py [q1|q2] [ratings.parquet]")
        sys.exit(1)

    question = sys.argv[1]
    ratings_path = sys.argv[2]
    output_folder = "results"

    if question == "q1":
        question_1_spark(ratings_path, output_folder)
    elif question == "q2":
        question_2_spark(ratings_path, output_folder)
    else:
        print("Invalid question. Choose 'q1' or 'q2'")
