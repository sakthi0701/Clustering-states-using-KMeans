# Clustering-states-using-KMeans  
Project Overview  
This project Clusters the states of India using the 2011 data of Literacy Rate, GDP per Capita, Population Growth Rate using the K-Means clustering algorithm. The goal was to explore how states group together based on features like GDP, literacy rate, and population growth. This was my first hands-on experience applying machine learning in Python.  


Tools & Technologies:  
Python  
pandas, scikit-learn, matplotlib  
Jupyter Notebook / VS Code  
Dataset sourced from data.gov.in  

Features Used for Clustering:  
GDP per capita(in ₹ crore/Population in crores)  
Population Growth Rate (Decadal %)  
Literacy Rate (%)  

Steps:  
Planning:I first asked ChatGPT for a step-by-step breakdown of the project.  
Environment Setup: Installed Python and relevant libraries since I had no prior experience with ML algorithms.  
Data Collection: I sourced the data from government-authorized website.  
Data Cleaning: Cleaned the data manually for consistency.  
Feature Scaling: Standardized the features using StandardScaler so that each attribute contributes equally to clustering.  
KMeans Clustering: Applied KMeans(n_clusters=3) to group the states.  
Visualization: Used PCA to reduce dimensions and visualized the clusters in 2D.  
Analysis: Interpreted the clusters by calculating mean values for each group.  
Here something was off,the result didn't make me happy so I went deeper and started thinking like that Domain expert which made me to make some changes in the attributes such as just GDP to GDP per capita which finally gave good results.


Observations & Inference:  
Cluster	Traits:  
2	States with high literacy and high GDP per capita but low population growth  
0	States that are average across all features — not high or low performers  
1	States that are low in GDP, low in literacy, and high in population growth  


Learnings:  
Learned to play with data and find actual useful results from them.  
Experienced the full ML project cycle: data collection → cleaning → modeling → interpretation.  
(https://github.com/user-attachments/assets/4497a2f4-7a53-46bd-a1f6-c10de02c145c)  
