import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("KMeansalgo/Clustered_States_Data.csv")

features = ["literacy", "growthrate", "gdp per capita"]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)

df["PC1"] = pca_result[:, 0]
df["PC2"] = pca_result[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="PC1", y="PC2", hue="Cluster", palette="Set1", s=120)

for i in range(len(df)):
    plt.text(df["PC1"][i] + 0.02, df["PC2"][i] + 0.02, df["state"][i], fontsize=9)

plt.title("Indian States Clustering (KMeans + PCA)", fontsize=14)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()
