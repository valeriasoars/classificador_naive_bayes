import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar a base já pré-processada e balanceada
df = pd.read_csv('dados_para_treino.csv')

# 2. Separar features (X) e target (y)
X = df.drop('order_status', axis=1)
y = df['order_status']

# 3. Configurar cross-validation estratificada
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4. Treinar e prever usando cross-validation
nb = GaussianNB()
y_pred = cross_val_predict(nb, X, y, cv=skf)

# 5. Avaliação: relatório e matriz de confusão
print(classification_report(y, y_pred, target_names=["Entregue", "Cancelado"]))
print(confusion_matrix(y, y_pred))

# 6. Gráfico da matriz de confusão
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Entregue", "Cancelado"],
            yticklabels=["Entregue", "Cancelado"])
plt.xlabel('Valores Previstos')
plt.ylabel('Valores Reais')
plt.title('Matriz de Confusão')
plt.tight_layout()
plt.show()
plt.savefig('matriz_confusao.png')
