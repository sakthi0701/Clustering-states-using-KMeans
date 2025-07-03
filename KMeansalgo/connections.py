import pandas as pd
df = pd.read_csv("KMeansalgo/Clustered_States_Data.csv")
cluster_profile = df.groupby("Cluster")[["literacy", "growthrate", "gdp per capita"]].mean()
print("Cluster Profiles (Averages):")
print(cluster_profile)
