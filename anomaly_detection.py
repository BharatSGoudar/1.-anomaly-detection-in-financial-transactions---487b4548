import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Step 1: Generate Synthetic Transaction Data
np.random.seed(42)
num_rows = 1000

amounts = np.random.normal(loc=100, scale=20, size=num_rows)
times = np.random.normal(loc=500, scale=100, size=num_rows)
account_ages = np.random.normal(loc=36, scale=10, size=num_rows)

# Inject anomalies
for _ in range(15):
    idx = np.random.randint(0, num_rows)
    amounts[idx] *= np.random.randint(5, 10)
    times[idx] += np.random.randint(1000, 2000)
    account_ages[idx] = np.random.randint(0, 5)

df = pd.DataFrame({
    'amount': amounts,
    'transaction_time': times,
    'account_age': account_ages
})

# Save to CSV
df.to_csv('transactions.csv', index=False)
print("✅ 'transactions.csv' file created.")

# Step 2: Load and Prepare Data
df = pd.read_csv('transactions.csv')
features = ['amount', 'transaction_time', 'account_age']
df_clean = df[features].dropna()

# Step 3: Normalize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

# Step 4: Train Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
df['anomaly'] = model.fit_predict(X_scaled)
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})  # 1 = anomaly

print(f"🔎 Total anomalies detected: {df['anomaly'].sum()}")

# Step 5: Visualize Anomalies
sns.set(style="whitegrid")
sns.scatterplot(data=df, x='amount', y='transaction_time', hue='anomaly', palette={0: 'blue', 1: 'red'})
plt.title("Detected Anomalies in Financial Transactions")
plt.xlabel("Transaction Amount")
plt.ylabel("Transaction Time")
plt.legend(title="Anomaly")
plt.tight_layout()
plt.savefig("anomaly_plot.png")
plt.show()

# Step 6: Save Output
df.to_csv('transactions_with_anomalies.csv', index=False)
print("✅ Results saved to 'transactions_with_anomalies.csv' and plot saved as 'anomaly_plot.png'.")
