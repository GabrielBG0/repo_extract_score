import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Configuration and Data Loading
FILE_NAME = 'extractability_analysis_results_v7.csv'
# Set a style for better aesthetics
sns.set_theme(style="whitegrid")

try:
    # Load the dataset
    df = pd.read_csv(FILE_NAME)
    print(f"Successfully loaded data from {FILE_NAME}. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: The file '{FILE_NAME}' was not found. Please ensure it's in the correct directory.")
    exit()
except Exception as e:
    print(f"An error occurred during file loading: {e}")
    exit()

# 2. Display Basic Information and Descriptive Statistics
print("\n--- Basic Dataset Information ---")
df.info()

print("\n--- Descriptive Statistics for Key Numerical Columns ---")
# Select numerical columns for a summary, focusing on the scores
key_scores = ['extractability_score', 'coupling_score', 'cohesion_score', 'modularity_score', 'complexity_score']
print(df[key_scores].describe())

# --- 3. Data Visualization with Seaborn ---

# Set up the figure for the visualizations
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Extractability Analysis Results Showcase', fontsize=16)

## Plot 1: Distribution of Extractability Score (Histogram/KDE)
# Show the distribution of the main outcome metric
sns.histplot(
    data=df,
    x='extractability_score',
    kde=True,
    bins=30,
    ax=axes[0],
    color='skyblue',
    edgecolor='black'
)
axes[0].set_title('Distribution of Extractability Scores')
axes[0].set_xlabel('Extractability Score')
axes[0].set_ylabel('Number of Repositories')
axes[0].axvline(df['extractability_score'].mean(), color='red', linestyle='--', label=f"Mean: {df['extractability_score'].mean():.2f}")
axes[0].legend()

## Plot 2: Relationship between Extractability and Coupling (Scatter Plot)
# Coupling generally reduces extractability, so we expect a negative correlation.
sns.scatterplot(
    data=df,
    x='coupling_score',
    y='extractability_score',
    hue='modularity_score',  # Use another relevant score for color scale
    size='complexity_score', # Use complexity to vary point size
    palette='viridis',
    sizes=(20, 200),
    alpha=0.7,
    ax=axes[1]
)
axes[1].set_title('Extractability Score vs. Coupling Score')
axes[1].set_xlabel('Coupling Score (Lower is better)')
axes[1].set_ylabel('Extractability Score (Higher is better)')
axes[1].legend(title='Modularity Score')

# Adjust layout to prevent overlapping titles/labels
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()