


``` python
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


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
# 4) KNN "From Scratch" (igual ao material)
# ------------------------------------------
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # Compute Euclidean distances
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        # Get indices of k-nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Get corresponding labels
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return majority class
        most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
        return most_common

# Treinar e avaliar
knn = KNNClassifier(k=3)  # k=3 como no exemplo da página
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, pred):.2f}")

```


``` python exec="on"
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


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
# 4) KNN "From Scratch" (igual ao material)
# ------------------------------------------
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # Compute Euclidean distances
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        # Get indices of k-nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Get corresponding labels
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return majority class
        most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
        return most_common

# Treinar e avaliar
knn = KNNClassifier(k=3)  # k=3 como no exemplo da página
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, pred):.2f}")
```


=== "Knn"

    ``` python exec="on" 
        --8<-- "docs/roteiro2/knn.py"
    ```