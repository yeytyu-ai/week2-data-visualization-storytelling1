import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output folders
os.makedirs("plots", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Load the dataset
file_path = "data/beauty_products.csv"

if not os.path.exists(file_path):
    print("Dataset not found!")
    print("Please add beauty_products.csv to the data folder.")
    raise SystemExit(1)

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nSummary statistics:")
print(df.describe(include="all").to_string())

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Convert common numeric columns if available
for column in ["price", "rating", "reviews"]:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

# 1. Bar chart: Product counts by category
if "category" in df.columns:
    df["category"].value_counts().head(10).plot(
        kind="bar", figsize=(10, 6)
    )
    plt.title("Top 10 Beauty Product Categories")
    plt.xlabel("Category")
    plt.ylabel("Number of Products")
    plt.tight_layout()
    plt.savefig("plots/category_bar_chart.png")
    plt.close()

# 2. Histogram: Price distribution
if "price" in df.columns:
    df["price"].dropna().plot(
        kind="hist", bins=20, figsize=(8, 5)
    )
    plt.title("Beauty Product Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("plots/price_histogram.png")
    plt.close()

# 3. Bar chart: Average rating by category
if "category" in df.columns and "rating" in df.columns:
    rating_data = df.groupby("category")["rating"].mean().dropna()
    rating_data.sort_values(ascending=False).head(10).plot(
        kind="bar", figsize=(10, 6)
    )
    plt.title("Top Categories by Average Rating")
    plt.xlabel("Category")
    plt.ylabel("Average Rating")
    plt.tight_layout()
    plt.savefig("plots/category_average_rating.png")
    plt.close()

# 4. Heatmap: Correlation between numerical columns
numeric_df = df.select_dtypes(include="number")

if numeric_df.shape[1] >= 2:
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("plots/correlation_heatmap.png")
    plt.close()

# Save summary information
df.describe(include="all").to_csv("output/summary_statistics.csv")

print("\nVisualization process completed.")
print("Check the plots folder for charts.")
