import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Unemployment in India.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Display dataset information
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# -----------------------------
# Graph 1: Unemployment Trend
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(df['Date'],
         df['Estimated Unemployment Rate (%)'],
         marker='o')

plt.title("Unemployment Trend in India")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# Graph 2: Average Unemployment by Region
# -----------------------------
region_avg = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean()

plt.figure(figsize=(12, 8))
region_avg.sort_values().plot(kind='barh')
plt.title("Average Unemployment Rate by Region")
plt.xlabel("Unemployment Rate (%)")
plt.tight_layout()
plt.show()

# -----------------------------
# Graph 3: COVID-19 Impact
# -----------------------------
covid_data = df[df['Date'] >= '2020-03-01']

plt.figure(figsize=(12, 6))
plt.plot(covid_data['Date'],
         covid_data['Estimated Unemployment Rate (%)'],
         marker='o')

plt.title("Impact of COVID-19 on Unemployment")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# Top 10 Regions with Highest Unemployment
# -----------------------------
print("\nTop 10 Regions by Average Unemployment Rate:")
print(region_avg.sort_values(ascending=False).head(10))

# -----------------------------
# Average Unemployment Rate
# -----------------------------
print("\nOverall Average Unemployment Rate:")
print(df['Estimated Unemployment Rate (%)'].mean())