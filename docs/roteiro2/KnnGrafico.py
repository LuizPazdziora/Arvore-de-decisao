import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# -------------------------------------------------------------------
# 1) Carregar base preprocessada
# -------------------------------------------------------------------
df = pd.read_csv(
    "C:/Users/lfpaz/OneDrive - ESPM/"
    "Sistema de Informação/SI4/MachineLearning/Customer-Churn-preprocessed.csv"
)

X = df.drop(columns=["Churn"])
y = df["Churn"]

# Padronizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -------------------------------------------------------------------
# 2) PCA → reduzir para 2 dimensões (para visualização)
# -------------------------------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# -------------------------------------------------------------------
# 3) Split treino/teste no espaço reduzido
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_pca,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------------------------------------------------
# 4) Treinar KNN
# -------------------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
print(f"Acurácia no espaço PCA: {accuracy_score(y_test, predictions):.3f}")


# -------------------------------------------------------------------
# 5) FRONTEIRA DE DECISÃO (MESMA LÓGICA DO EXEMPLO)
# -------------------------------------------------------------------
h = 0.02  # passo da malha

x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.figure(figsize=(12, 10))
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)

sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=y,
    style=y,
    palette="deep",
    s=80
)

plt.xlabel("Componente Principal 1 (PCA)")
plt.ylabel("Componente Principal 2 (PCA)")
plt.title("KNN Decision Boundary (Customer Churn após PCA)")

plt.savefig("C:/Users/lfpaz/OneDrive - ESPM/"
    "Sistema de Informação/SI4/MachineLearning/knn-pca-decision-boundary.svg", format="svg", transparent=True)
plt.close()
