import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler # Para balancear as classes via oversampling (duplicando exemplos da classe minoritária)

# 1. Carregar os arquivos 
encodings = {
    'utf-8': ['channels.csv', 'deliveries.csv', 'drivers.csv', 'orders.csv', 'payments.csv'],
    'latin1': ['hubs.csv', 'stores.csv']
}
dfs = {}

for encoding, arquivos in encodings.items():
    for arq in arquivos:
        nome = arq.split('.')[0]
        dfs[nome] = pd.read_csv(f'data/{arq}', encoding=encoding)

# 2. Merge das tabelas principais (usando orders como base)
df = dfs['orders'] \
    .merge(dfs['stores'], how='left', on='store_id') \
    .merge(dfs['hubs'], how='left', on='hub_id') \
    .merge(dfs['deliveries'], how='left', left_on='delivery_order_id', right_on='delivery_order_id') \
    .merge(dfs['channels'], how='left', on='channel_id') \
    .merge(dfs['payments'], how='left', left_on='payment_order_id', right_on='payment_order_id') \
    .merge(dfs['drivers'], how='left', left_on='driver_id', right_on='driver_id')

# 3. Remover IDs e colunas com todos valores únicos
drop_ids = ['order_id', 'delivery_order_id', 'payment_order_id', 'driver_id',
            'store_id', 'hub_id', 'payment_id', 'channel_id']
df = df.drop(columns=[col for col in drop_ids if col in df.columns])
df = df[[col for col in df.columns if df[col].nunique() != df.shape[0]]]

# 4. Tratamento de nulos
for col in df.select_dtypes('number').columns:
    # Preenche valores ausentes em colunas numéricas com a mediana
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes('object').columns:
     # Preenche strings/categóricas com "UNKNOWN", para que nada fique nulo e tudo seja codificável.
    df[col] = df[col].fillna('UNKNOWN')

# 5. Codificação categórica
onehot_cols = [
    'channel_type', 'delivery_status', 'driver_modal', 'driver_type',
    'hub_city', 'hub_state', 'store_segment', 'payment_method', 'payment_status'
]
onehot_cols = [c for c in onehot_cols if c in df.columns]
# Primeiro One-Hot nas recomendadas
df = pd.get_dummies(df, columns=onehot_cols)
#LabelEncoder nas categóricas remanescentes
le_dict = {}
for col in df.select_dtypes('object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# 6. Exclusão de colunas altamente correlacionadas
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95) and column != 'order_status']
df = df.drop(columns=to_drop)
print(f"Colunas removidas por correlação >0.95: {to_drop}")

# 7. Criação do alvo binário
if 'order_status' in df.columns:
    df['order_status'] = (df['order_status'] == le_dict['order_status'].transform(['CANCELED'])[0]).astype(int)

# 8. Separação de X e y
y = df['order_status']
X = df.drop(columns=['order_status'])

# 9. Balanceamento das classes
ros = RandomOverSampler(random_state=42)
X_bal, y_bal = ros.fit_resample(X, y)

# Cria um DataFrame novo, junta as features balanceadas e o alvo balanceado e salva como CSV.
df_bal = X_bal.copy()
df_bal['order_status'] = y_bal
df_bal.to_csv('dados_para_treino.csv', index=False)

print(f"Total de colunas finais após preprocessamento: {df_bal.shape[1]}")
print("Pré-processamento concluído! 'dados_para_treino.csv' pronto para uso no Naive Bayes.")

