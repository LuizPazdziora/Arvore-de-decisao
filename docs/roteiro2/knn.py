import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier

url = "https://raw.githubusercontent.com/LuizPazdziora/documentation.template/refs/heads/main/Customer-Churn.csv"
df = pd.read_csv(url)

# 2) Remover ID
df.drop(columns=["customerID"], inplace=True, errors="ignore")

df['gender'].fillna(df['gender'].mode(), inplace=True)
df['SeniorCitizen'].fillna(df['SeniorCitizen'].mode(), inplace=True)
df['Partner'].fillna(df['Partner'].mode(), inplace=True)
df['Dependents'].fillna(df['Dependents'].mode(), inplace=True)
df['tenure'].fillna(df['tenure'].mode()[0], inplace=True)
df['PhoneService'].fillna(df['PhoneService'].mode(), inplace=True)
df['MultipleLines'].fillna(df['MultipleLines'].mode(), inplace=True)
df['InternetService'].fillna(df['InternetService'].mode(), inplace=True)
df['OnlineSecurity'].fillna(df['OnlineSecurity'].mode(), inplace=True)
df['OnlineBackup'].fillna(df['OnlineBackup'].mode(), inplace=True)
df['DeviceProtection'].fillna(df['DeviceProtection'].mode()[0], inplace=True)
df['TechSupport'].fillna(df['TechSupport'].mode()[0], inplace=True)
df['StreamingTV'].fillna(df['StreamingTV'].mode()[0], inplace=True)
df['StreamingMovies'].fillna(df['StreamingMovies'].mode()[0], inplace=True)
df['Contract'].fillna(df['Contract'].mode()[0], inplace=True)
df['PaperlessBilling'].fillna(df['PaperlessBilling'].mode()[0], inplace=True)
df['PaymentMethod'].fillna(df['PaymentMethod'].mode()[0], inplace=True)
df['MonthlyCharges'].fillna(df['MonthlyCharges'].mode()[0], inplace=True)
df['TotalCharges'].fillna(df['TotalCharges'].mode()[0], inplace=True)
df['Churn'].fillna(df['Churn'].mode()[0], inplace=True)


# Tratamento de valores missmatch na coluna TotalCharges
s = df['TotalCharges'].astype(str).str.strip().replace({"2283.3"})
df['TotalCharges'] = pd.to_numeric(s, errors='coerce').astype('Float64')

df['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen'], errors='coerce').astype('Float64')
df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').astype('Float64')
df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').astype('Float64')


label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])
df['Partner'] = label_encoder.fit_transform(df['Partner'])
df['Dependents'] = label_encoder.fit_transform(df['Dependents'])
df['PhoneService'] = label_encoder.fit_transform(df['PhoneService'])
df['MultipleLines'] = label_encoder.fit_transform(df['MultipleLines'])
df['InternetService'] = label_encoder.fit_transform(df['InternetService'])
df['OnlineSecurity'] = label_encoder.fit_transform(df['OnlineSecurity'])
df['OnlineBackup'] = label_encoder.fit_transform(df['OnlineBackup'])
df['DeviceProtection'] = label_encoder.fit_transform(df['DeviceProtection'])
df['TechSupport'] = label_encoder.fit_transform(df['TechSupport'])
df['StreamingTV'] = label_encoder.fit_transform(df['StreamingTV'])
df['StreamingMovies'] = label_encoder.fit_transform(df['StreamingMovies'])
df['Contract'] = label_encoder.fit_transform(df['Contract'])
df['PaperlessBilling'] = label_encoder.fit_transform(df['PaperlessBilling'])
df['PaymentMethod'] = label_encoder.fit_transform(df['PaymentMethod'])


# Definir alvo (Churn) — se já virou 0/1 pelo LabelEncoder, só confirmamos o nome
# Definir alvo (Churn) — se já virou 0/1 pelo LabelEncoder, só confirmamos o nome  X e y 100% numéricos (elimina pd.NA e garante float)

target_col = "Churn"
X_df = df.drop(columns=[target_col])
X_df = X_df.apply(pd.to_numeric, errors='coerce')
X_df = X_df.fillna(X_df.median(numeric_only=True))
X = X_df.to_numpy(dtype=float)
y = df[target_col].to_numpy()
# Split treino/teste

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------
# 3) Split treino/teste (como no exemplo)
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

plt.figure(figsize=(12, 10))

# Generate synthetic dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")

# Visualize decision boundary
h = 0.02  # Step size in mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, style=y, palette="deep", s=100)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("KNN Decision Boundary (k=3)")

# Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())