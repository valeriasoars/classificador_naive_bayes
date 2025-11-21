import pandas as pd
import os

def ler_csv_robusto(caminho, sep=','):
    """
    Tenta ler um CSV usando utf-8, se falhar tenta latin1.
    Retorna o DataFrame ou None.
    """
    for encoding in ['utf-8', 'latin1']:
        try:
            df = pd.read_csv(caminho, encoding=encoding, sep=sep)
            print(f"Arquivo lido: {caminho} (encoding: {encoding})")
            return df
        except Exception as e:
            print(f'Falhou ao ler {caminho} com {encoding}: {e}')
    print(f"Atenção: arquivo {caminho} não pode ser lido.")
    return None

# Ajuste os nomes conforme seus arquivos
arquivos = {
    "channels": "channels.csv",
    "deliveries": "deliveries.csv",
    "drivers": "drivers.csv",
    "hubs": "hubs.csv",
    "orders": "orders.csv",
    "payments": "payments.csv",
    "stores": "stores.csv"
}

pasta = "data"
dfs = {}
for nome, arq in arquivos.items():
    caminho = os.path.join(pasta, arq)
    
    # Se necessário, ajuste o separador para ';' 
    sep = ','
    try:
        # Testa se a primeira linha tem muito ponto e vírgula
        with open(caminho, 'r', encoding='utf-8') as f:
            primeira = f.readline()
        if primeira.count(';') > primeira.count(','):
            sep = ';'
    except Exception as e:
        pass
    
    df = ler_csv_robusto(caminho, sep=sep)
    if df is not None:
        dfs[nome] = df

# Continua com análise: mostra nomes, tipos e 5 linhas de cada DataFrame
for nome, df in dfs.items():
    print(f"\n== {nome.upper()} ==")
    print("Shape:", df.shape)
    print("Colunas:", df.columns.tolist())
    print("Tipos:", df.dtypes)
    print("Nulos por coluna:", df.isnull().sum())
    print("Exemplo de linhas:")
    print(df.head())

# Verifica distribuição do status do pedido
if 'orders' in dfs and 'order_status' in dfs['orders']:
    print('\n== Distribuição do order_status (target) ==')
    print(dfs['orders']['order_status'].value_counts())
