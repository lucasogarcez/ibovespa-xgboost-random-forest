"""Executa o experimento completo (cenários A e B) e grava tudo em ./resultados/."""
import sys
from ibov_pipeline import Config, run_experiment, save_all, download_data

if __name__ == "__main__":
    cfg0 = Config()
    df = download_data(cfg0.ticker, cfg0.start, cfg0.end, cfg0.cache_csv)   # baixa uma vez e guarda em dados_bvsp.csv
    print(f"Dados: {df.index.min().date()} a {df.index.max().date()} ({len(df)} linhas)")
    for sc in ("B", "A"):
        cfg = Config(scenario=sc)
        res = run_experiment(cfg, df_raw=df, progress=print)
        save_all(res, f"resultados/cenario_{sc}")
        print(res["desempenho"].round(3))
        print({k: round(v["p_valor"], 4) for k, v in res["testes"].items()})
    print("\nPronto. Abra resultados/cenario_B/resultados_para_artigo.md")
