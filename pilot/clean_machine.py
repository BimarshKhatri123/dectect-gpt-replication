import pandas as pd
p = r"data\kaggle\machine_essays.csv"
df = pd.read_csv(p)
df.to_csv(r"data\kaggle\machine_essays_raw.csv", index=False)   # backup of the original
n0 = len(df)
df = df[~df["text"].str.contains(r"^[^.!?]{5,80}?:\s", regex=True)]            # drop essays that start with a title
df["text"] = df["text"].str.replace(r"\[[^\]]{2,30}\]", "", regex=True)         # remove [Your Name] placeholders
df["text"] = df["text"].str.split().str.join(" ")
df["word_count"] = df["text"].str.split().str.len()
df.to_csv(p, index=False)
print(f"kept {len(df)} of {n0}"); print(df.groupby("prompt_id").size())