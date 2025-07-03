import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("KMeansalgo/Combined_States_Data.csv")

features = ["literacy", "growthrate", "gdp per capita"]
X = df[features]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)

scaled_df = pd.DataFrame(scaled_features, columns=[f"{col}_scaled" for col in features])

if 'State' in df.columns:
    scaled_df['State'] = df['State']

if 'State' in df.columns:
    cols = ['State'] + [col for col in scaled_df.columns if col != 'State']
    scaled_df = scaled_df[cols]

scaled_df.to_csv("KMeansalgo/Scaled_States_Data.csv", index=False)

print("✅ Data has been scaled and saved to 'Scaled_States_Data.csv'")
