from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from io import StringIO


def preprocess(df):
    
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

    # Select features (suas colunas)
    features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges', 'Churn'
    ]

    return df[features]

url = "https://raw.githubusercontent.com/LuizPazdziora/documentation.template/refs/heads/main/Customer-Churn.csv"
df = pd.read_csv(url)


# Carregar o conjunto de dados

df = preprocess(df)


cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]
x = df[cols]
y = df['Churn']

# Dividir os dados em conjuntos de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Criar e treinar o modelo de árvore de decisão
classifier = tree.DecisionTreeClassifier()
classifier.fit(x_train, y_train)

# Evaluate the model
y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Optional: Print feature importances
feature_importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': classifier.feature_importances_
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)

print("<br>Feature Importances:")
print(feature_importance.to_html(index=False))

# Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())