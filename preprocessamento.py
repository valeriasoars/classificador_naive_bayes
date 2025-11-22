import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler # Para balancear as classes via oversampling (duplicando exemplos da classe minoritária)

# Etapa 1: Carregar os arquivos. Usa dois tipos de encoding, pois alguns arquivos são utf-8 e outros latin1 (para evitar erro de leitura).
encodings = {'utf-8': ['channels.csv', 'deliveries.csv', 'drivers.csv', 'orders.csv', 'payments.csv'],
             'latin1': ['hubs.csv', 'stores.csv']}
dfs = {}

# Salva os DataFrames num dicionário para acesso por nome.
for encoding, arquivos in encodings.items():
    for arq in arquivos:
        nome = arq.split('.')[0]
        dfs[nome] = pd.read_csv(f'data/{arq}', encoding=encoding)

# Etapa 2: Join das tabelas principais
# tabela orders é o DataFrame de partida, a base que você quer ampliar.
df = dfs['orders'] \
    .merge(dfs['stores'], how='left', on='store_id') \
    .merge(dfs['hubs'], how='left', on='hub_id') \
    .merge(dfs['deliveries'], how='left', left_on='delivery_order_id', right_on='delivery_order_id') \
    .merge(dfs['channels'], how='left', on='channel_id') \
    .merge(dfs['payments'], how='left', left_on='payment_order_id', right_on='payment_order_id') \
    .merge(dfs['drivers'], how='left', left_on='driver_id', right_on='driver_id')

# Etapa 3: Seleção e limpeza de features : Remove colunas de ID puro, que não carregam informação para classificação e podem atrapalhar o modelo
drop_ids = ['order_id', 'delivery_order_id', 'payment_order_id', 'driver_id', 'store_id', 'hub_id', 'payment_id', 'channel_id']
df = df.drop(columns=[col for col in drop_ids if col in df.columns])

# Etapa 4: Tratamento de nulos
for col in df.select_dtypes('number').columns:
    # Preenche valores ausentes em colunas numéricas com a mediana
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes('object').columns:
    # Preenche strings/categóricas com "UNKNOWN", para que nada fique nulo e tudo seja codificável.
    df[col] = df[col].fillna('UNKNOWN')

# Etapa 5: Codificação - OneHot para nominais com poucas categorias, LabelEncoder para o resto
onehot_cols = [
    'channel_type', 'delivery_status', 'driver_modal', 'driver_type',
    'hub_city', 'hub_state', 'store_segment', 'payment_method', 'payment_status'
]
onehot_cols = [c for c in onehot_cols if c in df.columns]  # só aplica se existe no dataframe

# Primeiro One-Hot nas recomendadas
df = pd.get_dummies(df, columns=onehot_cols)

# Depois LabelEncoder nas categóricas remanescentes
le_dict = {}
for col in df.select_dtypes('object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Etapa 6: Criação do alvo binário
if 'order_status' in df.columns:
    # Ajuste: 1 = CANCELADO, 0 = ENTREGUE (FINISHED)
    df['order_status'] = (df['order_status'] == le_dict['order_status'].transform(['CANCELED'])[0]).astype(int)

# Etapa 7: Separação de X e y
y = df['order_status']
X = df.drop(columns=['order_status'])

# Etapa 8: Balanceamento com oversample
# Duplica exemplos da classe “CANCELADO” (classe minoritária) até igualar o número de exemplos de cada classe.
# Balanceia a base para que o modelo aprenda a importância das duas classes igualmente, evitando viés para a maioria.
ros = RandomOverSampler(random_state=42)
X_bal, y_bal = ros.fit_resample(X, y)

# Cria um DataFrame novo, junta as features balanceadas e o alvo balanceado e salva como CSV.
df_bal = X_bal.copy()
df_bal['order_status'] = y_bal
df_bal.to_csv('dados_para_treino.csv', index=False)

print("Pré-processamento concluído! 'dados_para_treino.csv' pronto para uso no Naive Bayes.")

