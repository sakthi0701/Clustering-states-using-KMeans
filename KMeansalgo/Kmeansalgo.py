import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("KMeansalgo/Combined_States_Data.csv")

features = ["literacy", "growthrate", "gdp per capita"]
X = df[features]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled_features)

df.to_csv("KMeansalgo/Clustered_States_Data.csv", index=False)
print("✅ KMeans clustering complete! Data saved to 'Clustered_States_Data.csv'")
