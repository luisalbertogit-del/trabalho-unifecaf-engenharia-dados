"""
pipeline_ecommerce.py
=====================
Protótipo funcional do pipeline ELT para e-commerce.
Simula as três etapas centrais do fluxo:
  1. EXTRAÇÃO  — dados de banco relacional, API e CSV
  2. CARGA     — persistência na camada Bronze (Data Lake local)
  3. TRANSFORMAÇÃO — limpeza, enriquecimento e modelagem (Silver → Gold)

Requer: pandas, faker
Instalar: pip install pandas faker
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
import random
from faker import Faker

fake = Faker('pt_BR')
random.seed(42)

# ── Diretórios que simulam o Data Lake (Medallion) ───────────────────────
BASE_DIR   = "data_lake"
BRONZE_DIR = os.path.join(BASE_DIR, "bronze")
SILVER_DIR = os.path.join(BASE_DIR, "silver")
GOLD_DIR   = os.path.join(BASE_DIR, "gold")

for d in [BRONZE_DIR, SILVER_DIR, GOLD_DIR]:
    os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# ETAPA 1 — EXTRAÇÃO (simula fontes reais)
# ══════════════════════════════════════════════════════════════════════════

def extrair_banco_relacional(n=200):
    """
    Simula extração de tabela 'pedidos' de banco MySQL/PostgreSQL via Airbyte.
    Em produção: Airbyte usa conector JDBC e despeja JSONL no S3.
    """
    print("[EXTRACT] Banco relacional → pedidos...")
    status_opcoes = ['entregue', 'cancelado', 'em_transporte', 'processando']
    registros = []
    for i in range(1, n + 1):
        data_pedido = datetime.now() - timedelta(days=random.randint(0, 365))
        registros.append({
            "id_pedido":    i,
            "id_cliente":   random.randint(1, 80),
            "id_produto":   random.randint(1, 30),
            "quantidade":   random.randint(1, 5),
            "valor_unitario": round(random.uniform(29.9, 999.9), 2),
            "status":       random.choice(status_opcoes),
            "data_pedido":  data_pedido.strftime("%Y-%m-%d %H:%M:%S"),
            "canal_venda":  random.choice(['site', 'app', 'marketplace']),
            # Dado sujo intencional: ~5% com valor negativo (erro de sistema)
            "desconto":     round(random.uniform(-50, 0) if random.random() < 0.05
                                  else random.uniform(0, 0.3) * random.uniform(29.9, 999.9), 2),
        })
    return pd.DataFrame(registros)


def extrair_api_marketing(n=100):
    """
    Simula resposta de API do Google Ads / Meta Ads via Airbyte.
    Em produção: Airbyte conecta na API e despeja JSON no S3.
    """
    print("[EXTRACT] API marketing → campanhas...")
    campanhas = ['Black_Friday', 'Natal_2024', 'Dia_das_Maes', 'Relampago_Julho', 'Volta_as_Aulas']
    registros = []
    for i in range(1, n + 1):
        data = datetime.now() - timedelta(days=random.randint(0, 180))
        cliques = random.randint(100, 5000)
        registros.append({
            "id_campanha":   i,
            "nome_campanha": random.choice(campanhas),
            "plataforma":    random.choice(['Google Ads', 'Meta Ads']),
            "data":          data.strftime("%Y-%m-%d"),
            "impressoes":    random.randint(1000, 50000),
            "cliques":       cliques,
            "conversoes":    random.randint(0, int(cliques * 0.08)),
            "custo_brl":     round(random.uniform(500, 8000), 2),
            # Dado sujo intencional: ~8% com campo vazio
            "utm_source":   random.choice(['google', 'facebook', '', None, 'instagram'])
                             if random.random() > 0.08 else None,
        })
    return pd.DataFrame(registros)


def extrair_csv_clientes():
    """
    Simula leitura de arquivo CSV exportado de sistema legado.
    Em produção: arquivo CSV dropado em bucket S3 → Airbyte detecta e ingere.
    """
    print("[EXTRACT] Arquivo CSV → clientes...")
    registros = []
    for i in range(1, 81):
        registros.append({
            "id_cliente":   i,
            "nome":         fake.name(),
            "email":        fake.email() if random.random() > 0.04 else "INVALIDO@",
            "cpf":          fake.cpf(),
            "cidade":       fake.city(),
            "estado":       fake.estado_sigla(),
            "data_cadastro": (datetime.now() - timedelta(days=random.randint(30, 730))).strftime("%Y-%m-%d"),
            # Dado sujo: ~6% com segmento nulo
            "segmento":     random.choice(['premium', 'regular', 'novo']) if random.random() > 0.06 else None,
        })
    return pd.DataFrame(registros)


# ══════════════════════════════════════════════════════════════════════════
# ETAPA 2 — CARGA NA CAMADA BRONZE (dados brutos, sem transformação)
# ══════════════════════════════════════════════════════════════════════════

def carregar_bronze(df, nome_tabela):
    """
    Persiste os dados brutos no Data Lake camada Bronze.
    Em produção: Airbyte grava diretamente no S3 em formato Parquet ou JSONL.
    """
    caminho = os.path.join(BRONZE_DIR, f"{nome_tabela}.parquet")
    df.to_parquet(caminho, index=False)
    print(f"  [BRONZE] {nome_tabela}: {len(df)} registros → {caminho}")
    return caminho


# ══════════════════════════════════════════════════════════════════════════
# ETAPA 3 — TRANSFORMAÇÃO: BRONZE → SILVER (limpeza e validação)
# ══════════════════════════════════════════════════════════════════════════

def transformar_pedidos_silver():
    """
    Limpeza e validação dos pedidos.
    Em produção: executado pelo AWS Glue (PySpark) ou dbt.
    """
    print("[TRANSFORM] Pedidos Bronze → Silver...")
    df = pd.read_parquet(os.path.join(BRONZE_DIR, "pedidos.parquet"))
    total_antes = len(df)

    # 1. Corrigir descontos negativos (erro de sistema)
    df['desconto'] = df['desconto'].clip(lower=0)

    # 2. Calcular valor total do item
    df['valor_total'] = (df['valor_unitario'] * df['quantidade']) - df['desconto']

    # 3. Converter data para tipo datetime
    df['data_pedido'] = pd.to_datetime(df['data_pedido'])
    df['ano_mes']     = df['data_pedido'].dt.to_period('M').astype(str)

    # 4. Remover registros com valor_total inválido (< 0)
    df = df[df['valor_total'] > 0]

    # 5. Padronizar status
    df['status'] = df['status'].str.lower().str.strip()

    total_depois = len(df)
    print(f"  Registros removidos (inválidos): {total_antes - total_depois}")

    caminho = os.path.join(SILVER_DIR, "pedidos.parquet")
    df.to_parquet(caminho, index=False)
    print(f"  [SILVER] pedidos: {total_depois} registros → {caminho}")
    return df


def transformar_clientes_silver():
    """
    Limpeza de clientes + mascaramento de PII (LGPD).
    Em produção: AWS Glue com mascaramento antes de chegar à camada analítica.
    """
    print("[TRANSFORM] Clientes Bronze → Silver...")
    df = pd.read_parquet(os.path.join(BRONZE_DIR, "clientes.parquet"))

    # 1. Validar email (remove claramente inválidos)
    df = df[df['email'].str.contains('@', na=False)]
    df = df[~df['email'].str.endswith('@')]

    # 2. Preencher segmento nulo com 'regular'
    df['segmento'] = df['segmento'].fillna('regular')

    # 3. Mascaramento PII (LGPD) — CPF e email parcialmente ocultados
    df['cpf_masked']   = df['cpf'].str[:3] + '.***.***-**'
    df['email_masked'] = df['email'].apply(
        lambda e: e[:2] + '***@' + e.split('@')[-1] if '@' in e else '***'
    )

    # 4. Manter colunas não sensíveis para análise
    df_silver = df[['id_cliente', 'cidade', 'estado', 'segmento',
                    'data_cadastro', 'cpf_masked', 'email_masked']]

    caminho = os.path.join(SILVER_DIR, "clientes.parquet")
    df_silver.to_parquet(caminho, index=False)
    print(f"  [SILVER] clientes: {len(df_silver)} registros → {caminho}")
    return df_silver


def transformar_campanhas_silver():
    """Limpeza das campanhas de marketing."""
    print("[TRANSFORM] Campanhas Bronze → Silver...")
    df = pd.read_parquet(os.path.join(BRONZE_DIR, "campanhas.parquet"))

    df['utm_source'] = df['utm_source'].fillna('desconhecido')
    df['data']       = pd.to_datetime(df['data'])
    df['ctr']        = (df['cliques'] / df['impressoes']).round(4)
    df['cpc_brl']    = (df['custo_brl'] / df['cliques'].replace(0, 1)).round(2)
    df['cpa_brl']    = (df['custo_brl'] / df['conversoes'].replace(0, 1)).round(2)

    caminho = os.path.join(SILVER_DIR, "campanhas.parquet")
    df.to_parquet(caminho, index=False)
    print(f"  [SILVER] campanhas: {len(df)} registros → {caminho}")
    return df


# ══════════════════════════════════════════════════════════════════════════
# ETAPA 4 — MODELAGEM: SILVER → GOLD (tabelas analíticas / DW)
# ══════════════════════════════════════════════════════════════════════════

def gerar_fato_vendas_gold():
    """
    Gera tabela fato de vendas (esquema estrela).
    Em produção: dbt materializa como tabela no Amazon Redshift.
    """
    print("[GOLD] Gerando fato_vendas...")
    pedidos  = pd.read_parquet(os.path.join(SILVER_DIR, "pedidos.parquet"))
    clientes = pd.read_parquet(os.path.join(SILVER_DIR, "clientes.parquet"))

    fato = pedidos.merge(clientes, on='id_cliente', how='left')
    fato = fato[fato['status'] == 'entregue']  # apenas vendas concluídas

    fato_gold = fato[[
        'id_pedido', 'id_cliente', 'id_produto', 'canal_venda',
        'quantidade', 'valor_total', 'data_pedido', 'ano_mes',
        'estado', 'segmento'
    ]].copy()
    fato_gold.rename(columns={'data_pedido': 'dt_venda'}, inplace=True)

    caminho = os.path.join(GOLD_DIR, "fato_vendas.parquet")
    fato_gold.to_parquet(caminho, index=False)
    print(f"  [GOLD] fato_vendas: {len(fato_gold)} registros → {caminho}")
    return fato_gold


def gerar_kpis_gold(fato_vendas):
    """
    Gera tabela de KPIs mensais agregados.
    Em produção: view materializada no Redshift, consumida pelo Power BI.
    """
    print("[GOLD] Gerando kpis_mensais...")
    kpis = fato_vendas.groupby('ano_mes').agg(
        total_pedidos   = ('id_pedido',   'count'),
        receita_total   = ('valor_total', 'sum'),
        ticket_medio    = ('valor_total', 'mean'),
        itens_vendidos  = ('quantidade',  'sum'),
    ).reset_index()
    kpis['receita_total'] = kpis['receita_total'].round(2)
    kpis['ticket_medio']  = kpis['ticket_medio'].round(2)
    kpis = kpis.sort_values('ano_mes')

    caminho = os.path.join(GOLD_DIR, "kpis_mensais.parquet")
    kpis.to_parquet(caminho, index=False)
    print(f"  [GOLD] kpis_mensais: {len(kpis)} períodos → {caminho}")
    return kpis


def gerar_ranking_produtos_gold(fato_vendas):
    """Ranking de produtos por receita — alimenta dashboard de vendas."""
    print("[GOLD] Gerando ranking_produtos...")
    rank = fato_vendas.groupby('id_produto').agg(
        total_vendas  = ('id_pedido',   'count'),
        receita       = ('valor_total', 'sum'),
        qtd_itens     = ('quantidade',  'sum'),
    ).reset_index().sort_values('receita', ascending=False)
    rank['receita'] = rank['receita'].round(2)

    caminho = os.path.join(GOLD_DIR, "ranking_produtos.parquet")
    rank.to_parquet(caminho, index=False)
    print(f"  [GOLD] ranking_produtos: {len(rank)} produtos → {caminho}")
    return rank


# ══════════════════════════════════════════════════════════════════════════
# RELATÓRIO DE QUALIDADE (simula Great Expectations)
# ══════════════════════════════════════════════════════════════════════════

def relatorio_qualidade(df_pedidos_bronze, df_pedidos_silver):
    """
    Simula validações automáticas de qualidade (Great Expectations).
    Em produção: suite de expectations executada pelo Airflow após cada carga.
    """
    print("\n[QUALITY] Relatório de Qualidade dos Dados")
    print("=" * 55)
    checks = {
        "Pedidos sem valor_total negativo": (df_pedidos_silver['valor_total'] >= 0).all(),
        "Status preenchido (sem nulos)":    df_pedidos_silver['status'].notna().all(),
        "Data pedido é datetime válido":    pd.api.types.is_datetime64_any_dtype(df_pedidos_silver['data_pedido']),
        "Registros removidos Bronze→Silver": len(df_pedidos_bronze) > len(df_pedidos_silver),
        "Canal de venda sem nulos":         df_pedidos_silver['canal_venda'].notna().all(),
    }
    for check, resultado in checks.items():
        status = "✓ PASS" if resultado else "✗ FAIL"
        print(f"  {status}  {check}")
    print("=" * 55)


# ══════════════════════════════════════════════════════════════════════════
# EXECUÇÃO DO PIPELINE COMPLETO
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  PIPELINE ELT — E-COMMERCE  |  Início:", datetime.now().strftime("%H:%M:%S"))
    print("="*55 + "\n")

    # ── EXTRACT ──────────────────────────────────────────────────────────
    print("── FASE 1: EXTRAÇÃO ─────────────────────────────────")
    df_pedidos   = extrair_banco_relacional(200)
    df_campanhas = extrair_api_marketing(100)
    df_clientes  = extrair_csv_clientes()

    # ── LOAD BRONZE ──────────────────────────────────────────────────────
    print("\n── FASE 2: CARGA BRONZE ────────────────────────────")
    carregar_bronze(df_pedidos,   "pedidos")
    carregar_bronze(df_campanhas, "campanhas")
    carregar_bronze(df_clientes,  "clientes")

    # ── TRANSFORM SILVER ─────────────────────────────────────────────────
    print("\n── FASE 3: TRANSFORMAÇÃO SILVER ────────────────────")
    df_ped_silver = transformar_pedidos_silver()
    transformar_clientes_silver()
    transformar_campanhas_silver()

    # ── QUALITY CHECK ─────────────────────────────────────────────────────
    relatorio_qualidade(df_pedidos, df_ped_silver)

    # ── GOLD / DW ────────────────────────────────────────────────────────
    print("\n── FASE 4: MODELAGEM GOLD (Data Warehouse) ─────────")
    fato_vendas = gerar_fato_vendas_gold()
    kpis        = gerar_kpis_gold(fato_vendas)
    gerar_ranking_produtos_gold(fato_vendas)

    # ── PREVIEW DOS KPIs ─────────────────────────────────────────────────
    print("\n── RESULTADO FINAL: KPIs Mensais (amostra) ─────────")
    print(kpis.tail(6).to_string(index=False))

    print("\n" + "="*55)
    print("  PIPELINE CONCLUÍDO:", datetime.now().strftime("%H:%M:%S"))
    print("="*55 + "\n")
