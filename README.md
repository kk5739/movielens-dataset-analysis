# MovieLens Dataset Analysis: Recommender Systems & Segmentation

## Overview

This project presents a comprehensive movie recommendation system and user segmentation analysis using the [MovieLens](https://grouplens.org/datasets/movielens/latest/) dataset. It was developed as a capstone project for NYU's **DSGA1004 - Big Data** course. The pipeline leverages PySpark, ALS, MinHashLSH, and modern evaluation metrics to address scalability and personalization in recommender systems.

## Contributors

- **Kyeongmo Kang** 
- **Nikolas Prasinos**
- **Alexander Pegot-Ogier** 


## Project Structure

```
movielens-dataset-analysis/
├── als_model.py               # Trains and evaluates ALS collaborative filtering model (Q5)
├── customer_segmentation.py  # Finds similar users using MinHash LSH and validates with Pearson correlation (Q1, Q2)
├── parquet_convertor.py      # Converts original MovieLens CSVs to optimized Parquet format
├── popularity_model.py       # Implements popularity-based recommender system as baseline (Q4)
├── split.py                  # Prepares training, validation, and test sets by timestamp (Q3)
├── ml-latest-small/          # MovieLens small dataset (CSV files)
├── results/                  # Top 100 user pairs, Q1/Q2 outputs
├── Report.pdf                # Full technical and analytical write-up
├── README.md                 # Project overview (this file)
├── requirements.txt          # Project dependencies
├── venv/                     # (Optional) Local virtual environment
```

## Objectives

- Identify "movie-twin" users with overlapping watch patterns using **MinHashLSH**
- Validate similarities through **Pearson correlation** of ratings
- Implement and benchmark a **popularity-based baseline recommender**
- Build and tune a **Spark ALS model** for collaborative filtering
- Evaluate all models using **Precision\@100**, **MAP\@100**, and **NDCG\@100**

## Datasets

- Source: [MovieLens](https://grouplens.org/datasets/movielens/latest/)
- Files used: `ratings.csv`, `movies.csv`, `tags.csv`, etc.
- Transformed to Parquet format for optimized Spark performance

## Deliverables

### 1. Customer Segmentation via MinHashLSH

- Converted ratings to sparse vectors
- Used 5 hash tables, 2500+ rating filter
- Identified top 100 most overlapping user pairs ("movie twins")

### 2. Rating Validation via Pearson Correlation

- Compared Jaccard-similar pairs with random user pairs
- Found: **Movie-twins Pearson avg = 0.7885**, random avg = 0.1523

### 3. Train/Validation/Test Splitting

- User-wise splits preserving timestamp order
- Filter: min 500 ratings/user
- Split: 80% Train / 10% Val / 10% Test

### 4. Popularity-Based Baseline Model

- Top-100 movies globally ranked from training set
- Evaluation on validation/test users using ranking metrics
- **Test Precision\@100: 0.0729**, **NDCG\@100: 0.0936**

### 5. ALS Collaborative Filtering

- Grid search over rank, regParam, maxIter
- Best config: `rank=100`, `regParam=0.05`, `maxIter=20`
- **Test RMSE = 0.7345**, **NDCG\@100 = 0.0137**

## Technologies Used

- Python 3.x
- Apache Spark (PySpark)
- pandas, NumPy, scikit-learn
- datasketch (for MinHash)

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Pipelines

### 1. Convert CSVs to Parquet

```bash
python parquet_convertor.py
```

### 2. Segment Users

```bash
python customer_segmentation.py
```

### 3. Validate with Pearson

(Part of `customer_segmentation.py`)

### 4. Split Ratings

```bash
python split.py
```

### 5. Baseline Recommender

```bash
python popularity_model.py
```

### 6. ALS Recommender

```bash
python als_model.py
```

## Evaluation Metrics

| Model            | Precision\@100 | MAP\@100 | NDCG\@100 | RMSE   |
| ---------------- | -------------- | -------- | --------- | ------ |
| Popularity Model | 0.0729         | 0.0136   | 0.0936    | -      |
| ALS Model        | 0.0119         | 0.0009   | 0.0137    | 0.7345 |

## Contributions

- **Kyeongmo Kang**
- **Alexander Pegot-Ogier**
- **Nikolas Prasinos**

## License

Academic use only. NYU DSGA1004 Capstone, Spring 2025

---

For detailed results, see [`Report.pdf`](./Report.pdf) and [`results/`](./results/)

