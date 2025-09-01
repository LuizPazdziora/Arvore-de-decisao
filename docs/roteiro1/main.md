## Objetivo

O dataset Customer Churn – reúne dados de clientes de uma operadora de telecom com o objetivo: prever se o cliente vai cancelar (Churn) ou permanecer.


## Descrição das colunas (dicionário de dados)
customerID: ID do cliente

Gender: Gênero do cliente (feminino, masculino)

SeniorCitizen: Se o cliente é idoso ou não (1, 0)


Partner: Se o cliente tem cônjuge/parceiro (Sim, Não)

Dependents: Se o cliente possui dependentes (Sim, Não)

Tenure: Número de meses que o cliente permaneceu na empresa

PhoneService: Se o cliente possui serviço de telefonia (Sim, Não)

MultipleLines: Se o cliente possui múltiplas linhas (Sim, Não, Sem serviço de telefonia)

InternetService: Provedor de internet do cliente (DSL, Fibra óptica, Sem internet)

OnlineSecurity: Se o cliente possui segurança online (Sim, Não, Sem serviço de internet)

OnlineBackup: Se o cliente possui backup online (Sim, Não, Sem serviço de internet)

DeviceProtection: Se o cliente possui proteção de dispositivos (Sim, Não, Sem serviço de internet)

TechSupport: Se o cliente possui suporte técnico (Sim, Não, Sem serviço de internet)

StreamingTV: Se o cliente possui streaming de TV (Sim, Não, Sem serviço de internet)

StreamingMovies: Se o cliente possui streaming de filmes (Sim, Não, Sem serviço de internet)

Contract: Tipo de contrato do cliente (Mês a mês, Um ano, Dois anos)

PaperlessBilling: Se o cliente utiliza fatura digital (Sim, Não)

PaymentMethod: Método de pagamento do cliente (Cheque eletrônico, Cheque enviado, Transferência bancária — automática, Cartão de crédito — automático)

MonthlyCharges: Valor cobrado mensalmente do cliente

TotalCharges: Valor total cobrado do cliente

Churn: Se o cliente cancelou (Sim ou Não)



## Pré Processamento


1) padronização de tipos

Normalização de TotalCharges para numérico

2) Tratamento de valores faltantes

Numéricos (tenure, MonthlyCharges, TotalCharges): imputação pela mediana.

Categóricos (gender, Partner, Dependents, PhoneService, MultipleLines, InternetService,
OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
Contract, PaperlessBilling, PaymentMethod, Churn): imputação pela moda (valor mais frequente).

``` python  exec='on' html='0'
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

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

    return df

url = "https://raw.githubusercontent.com/LuizPazdziora/documentation.template/refs/heads/main/Customer-Churn.csv"
df = pd.read_csv(url)
df = df.sample(n=10)
df = preprocess(df)

print(df.to_markdown(index=False))
```

## Divisão de dados

Os dados foram divididos em 70% para treino e 30% para validação, com o objetivo de evitar overfitting e obter uma estimativa mais fiel de desempenho.

``` python exec="on" html="on"
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
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


# Optional: Print feature importances
feature_importance = pd.DataFrame({
    'Feature': [ 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'],
    'Importance': classifier.feature_importances_
})
print("<br>Feature Importances:")
print(feature_importance.sort_values(by='Importance', ascending=False).to_html())

# Avaliar o modelo
accuracy = classifier.score(x_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
tree.plot_tree(classifier)

# Para imprimir na página HTML
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())


```




## Avaliação do modelo

O modelo teve accuracy de 74% no conjunto de validação. Para um accuracy maior é necessario análise das variáveis com maior correlação para serem utilizadas no treinamento do modelo


