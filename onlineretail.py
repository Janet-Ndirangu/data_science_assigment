# Import the required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
# Loading the dataset
data = pd.read_csv("OnlineRetail.csv", encoding="ISO-8859-1")
# first five rows of the dataset
print(data.head())
# the shape of the dataset
print("\nShape of the dataset: ", data.shape)
# column names
print("\nColumns: ", data.columns)
# Dataset information
print("\nDataset Information: ")
print(data.info())
# Dataset description: mean, count, std, variance, IQR, etc
print("\nDescription: \n", data.describe())
# Null values
print("\nNull Values in percentages: \n", round(100 * (data.isnull().sum())/len(data), 2))
# dropping rows that have missing values
data = data.dropna()
print("\nShape: ", data.shape)
"""data.drop(["StockCode"], axis=1, inplace=True)
print(data.columns)"""
# change the datatype of Customer ID as per business understanding
data["CustomerID"] = data["CustomerID"].astype(str)
print("\nData Information Two after changing CustomerID column to str:")
print(data.info())

# Data preparation
# New attribute: Monetary
"""
We are going to analysis the Customers based on below 3 factors:¶
R (Recency): Number of days since last purchase
F (Frequency): Number of transactions
M (Monetary): Total amount of transactions (revenue contributed)
"""
# Create a new column Amount that multiplies Quantity with UnitPrice
data["Amount"] = data['Quantity'] * data['UnitPrice']
# New Attribute: Monetary
data_monetary = data.groupby('CustomerID')['Amount'].sum().sort_values(ascending=False)
print(data_monetary.head())
# Identify the most sold product
# most sold product using the description and amount columns
most_sold_product_data = (data.groupby('Description')['Quantity'].sum().sort_values
                          (ascending=False).head())
print("\n", most_sold_product_data)
# Country that bought more products based on Country and Amount columns
most_purchases_country_data = (data.groupby('Country')['Amount'].sum().sort_values
                               (ascending=False).head())
print("\n", most_purchases_country_data)

print("\nMost sold product details:\n", most_sold_product_data.head(1))
print("\nCountry with most purchase details:\n", most_purchases_country_data.head(1))

data_monetary = data_monetary.reset_index()
print(data_monetary.head())

# frequency
data_frequency = (data.groupby('Description')["InvoiceNo"].count().sort_values
                  (ascending=False))
print("\n", data_frequency)

# the last month sold how much sales
# Convert 'InvoiceDate' to datetime datatype
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%m/%d/%Y %H:%M')
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
print(data["InvoiceDate"].tail())
# compute the maximum date to know the last transaction date
max_date = data['InvoiceDate'].max()
min_date = data['InvoiceDate'].min()
print("\nMinimum Date: ", min_date)
print("\nMaximum Date: ", max_date)
print("\nDifference: ", (max_date - min_date))
# compute the first day of the last month
start_date = max_date - pd.Timedelta(days=30)
last_30_days_data = data[data['InvoiceDate'] >= start_date]
# Format the dates
formatted_first_date_last_month = start_date.strftime('%Y-%m-%d')
formatted_last_date = max_date.strftime('%Y-%m-%d')
# Calculate total sales
total_sales_last_30_days = last_30_days_data['Amount'].sum()
# print or display the total sales within the last 30 days
print(f"\nTotal sales in the last 30 days (from {formatted_first_date_last_month}"
      f" to {formatted_last_date}): {total_sales_last_30_days}")


# Scale the data
scaler = StandardScaler()
data_monetary_scaled = scaler.fit_transform(data_monetary[['Amount']])

# Define the range for the number of clusters
K = range(2, 10)

# Calculate Silhouette Scores for the range of K and store the KMeans models
silhouette_scores = []
models = {}
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_monetary_scaled)
    score = silhouette_score(data_monetary_scaled, kmeans.labels_)
    silhouette_scores.append(score)
    models[k] = kmeans
print("\nScores: ", silhouette_scores)

# Plotting the Silhouette Scores
plt.figure(figsize=(10, 6))
plt.plot(K, silhouette_scores, 'bo-')  # 'bo-' means blue color, circle markers, solid line
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal values of K')
plt.show()

# Optimal K based on the highest silhouette score
# Optimal value of K is where the score appears to be the highest
optimal_k = K[silhouette_scores.index(max(silhouette_scores))]
print(f'\nThe optimal value of K is {optimal_k}')

# Visualizing the clusters for the optimal k (Which is two in our case)
kmeans_optimal = models[optimal_k]
data_monetary['Cluster'] = kmeans_optimal.labels_

plt.figure(figsize=(10, 6))
for cluster in range(optimal_k):
    cluster_data = data_monetary[data_monetary['Cluster'] == cluster]
    plt.scatter(cluster_data.index, cluster_data['Amount'], label=f'Cluster {cluster}')

plt.xlabel('Customer ID Index')
plt.ylabel('Monetary Value')
plt.title(f'Customer Clusters for k = {optimal_k}')
plt.legend()
plt.show()
