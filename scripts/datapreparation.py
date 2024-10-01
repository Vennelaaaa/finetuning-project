import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv(r"C:\Users\vennela\Downloads\my_llm_fine_tuning_project\data\impression_300_llm.csv")

# Split the dataset into train (300) and evaluation (30)
train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)

# Save the splits
train_df.to_csv("datasets/train_data.csv", index=False)
eval_df.to_csv("datasets/eval_data.csv", index=False)
