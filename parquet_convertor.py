from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
import sys
import os

def convert_csv_to_parquet(csv_path, parquet_path=None):
    """
    Convert a CSV file to Parquet format.
    
    Parameters:
    - csv_path: Path to the input CSV file
    - parquet_path: Path for the output Parquet file (optional)
                    If not provided, will use csv_path with .parquet extension
    """
    # If parquet_path is not provided, use the same path with .parquet extension
    if parquet_path is None:
        parquet_path = os.path.splitext(csv_path)[0] + '.parquet'
    
    # Create a Spark session
    spark = SparkSession.builder \
        .appName("ConvertToParquet") \
        .getOrCreate()
    
    try:
        # Define the schema for the ratings data
        schema = StructType([
            StructField("userId", IntegerType(), True),
            StructField("movieId", IntegerType(), True),
            StructField("rating", FloatType(), True),
            StructField("timestamp", IntegerType(), True)
        ])
        
        print(f"Reading CSV from: {csv_path}")
        
        # Read the CSV file with the defined schema
        ratings_df = spark.read.csv(csv_path, header=True, schema=schema)
        
        print(f"Writing Parquet to: {parquet_path}")
        
        # Write the data as Parquet with snappy compression
        ratings_df.write.mode("overwrite").option("compression", "snappy").parquet(parquet_path)
        
        print(f"✅ Converted and saved: {csv_path} → {parquet_path}")
        
        # Show record count
        count = ratings_df.count()
        print(f"Number of records: {count}")
        
        # Get and display file sizes
        try:
            # For HDFS paths
            if csv_path.startswith("hdfs://"):
                fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
                csv_size = fs.getContentSummary(spark._jvm.org.apache.hadoop.fs.Path(csv_path)).getLength()
                parquet_size = fs.getContentSummary(spark._jvm.org.apache.hadoop.fs.Path(parquet_path)).getLength()
            else:
                # For local file system
                csv_size = os.path.getsize(csv_path)
                # For parquet directory (approximate)
                parquet_size = sum(os.path.getsize(os.path.join(parquet_path, f)) 
                                  for f in os.listdir(parquet_path) 
                                  if f.endswith('.parquet') or f.endswith('.snappy.parquet'))
            
            print(f"CSV size: {csv_size / (1024 * 1024):.2f} MB")
            print(f"Parquet size: {parquet_size / (1024 * 1024):.2f} MB")
            print(f"Compression ratio: {csv_size / parquet_size:.2f}x")
        except Exception as e:
            print(f"Could not calculate file sizes: {str(e)}")
    
    finally:
        # Stop Spark session
        spark.stop()
        
    return parquet_path

if __name__ == "__main__":
    # Print usage if not enough arguments
    if len(sys.argv) < 2:
        print("Usage: spark-submit convert_to_parquet.py <csv_path> [parquet_path]")
        sys.exit(1)
    
    # Get CSV path from command line
    csv_path = sys.argv[1]
    
    # Get Parquet path from command line (if provided)
    parquet_path = None
    if len(sys.argv) >= 3:
        parquet_path = sys.argv[2]
    
    # Convert CSV to Parquet
    convert_csv_to_parquet(csv_path, parquet_path)