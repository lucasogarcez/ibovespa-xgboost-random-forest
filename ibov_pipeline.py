"""
Pipeline: Random Forest x XGBoost na direção diária do Ibovespa (^BVSP).

Fluxo: coleta (yfinance) -> limpeza -> atributos -> divisão cronológica 60/20/20
       -> busca em grade na validação -> reajuste em treino+validação
       -> avaliação única no teste -> McNemar -> figuras e texto de resultados.

Convenção temporal (evita vazamento):
  * Todo atributo da linha t usa apenas informação disponível até o fechamento de t.
  * O alvo da linha t é 1 se Fechamento[t+1] > Fechamento[t], senão 0.
  * ret_1 = retorno do pregão t (o último conhecido), ret_2 = pregão t-1, ret_3 = pregão t-2.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import ParameterGrid

# --------------------------------------------------------------------------
# Configuração
# --------------------------------------------------------------------------


@dataclass
class Config:
    ticker: str = "^BVSP"
    start: str = "2010-01-01"
    end: str = "2025-12-31"          # inclusivo
    scenario: str = "B"              # "A" = preços brutos (projeto original); "B" = atributos sem escala
    train_frac: float = 0.60
    val_frac: float = 0.20
    seed: int = 42
    n_seeds_robustez: int = 10
    cache_csv: str | None = "dados_bvsp.csv"
    rf_grid: dict = field(default_factory=lambda: {
        "n_estimators": [100, 300],
        "max_depth": [3, 5, 8, None],
    })
    xgb_grid: dict = field(default_factory=lambda: {
        "n_estimators": [100, 300],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.05, 0.1],
    })


# --------------------------------------------------------------------------
# Coleta e limpeza
# --------------------------------------------------------------------------


def download_data(ticker: str, start: str, end: str, cache_csv: str | None = None) -> pd.DataFrame:
    """Baixa OHLCV do Yahoo Finance. Se cache_csv existir, usa o arquivo (reprodutibilidade)."""
    if cache_csv and os.path.exists(cache_csv):
        return pd.read_csv(cache_csv, index_col=0, parse_dates=True)
    import yfinance as yf

    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance: fim exclusivo
    df = yf.download(ticker, start=start, end=end_excl, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    if cache_csv:
        df.to_csv(cache_csv)
    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Ordena, remove duplicatas, ausentes e preços não positivos. Retorna df e relatório."""
    rep = {"linhas_brutas": int(len(df))}
    df = df.sort_index()
    rep["duplicatas_removidas"] = int(df.index.duplicated().sum())
    df = df[~df.index.duplicated(keep="first")]
    ohlc = ["Open", "High", "Low", "Close"]
    rep["ausentes_removidos"] = int(df[ohlc].isna().any(axis=1).sum())
    df = df.dropna(subset=ohlc)
    nonpos = (df[ohlc] <= 0).any(axis=1)
    rep["precos_nao_positivos_removidos"] = int(nonpos.sum())
    df = df[~nonpos]
    rep["linhas_finais"] = int(len(df))
    return df, rep


# --------------------------------------------------------------------------
# Atributos
# --------------------------------------------------------------------------


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI de Wilder (suavização exponencial com alpha = 1/n). Usa apenas dados passados."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_down = down.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_up / avg_down.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out[(avg_down == 0.0) & avg_up.notna()] = 100.0
    return out


def build_features(df: pd.DataFrame, scenario: str = "B"):
    """Retorna (X, y, X_ultimo). X_ultimo = atributos do último pregão (sem alvo conhecido)."""
    if scenario not in ("A", "B"):
        raise ValueError("scenario deve ser 'A' ou 'B'")
    c, o, h, l = df["Close"], df["Open"], df["High"], df["Low"]
    ret = c.pct_change()
    f = pd.DataFrame(index=df.index)
    for k in (1, 2, 3):
        r = ret.shift(k - 1)
        f[f"ret_{k}"] = r
        f[f"dir_{k}"] = (r > 0).astype(float).where(r.notna())
    sma5, sma10 = c.rolling(5).mean(), c.rolling(10).mean()
    if scenario == "A":      # projeto original: níveis de preço
        f["open"], f["high"], f["low"], f["close"] = o, h, l, c
        f["sma_5"], f["sma_10"] = sma5, sma10
    else:                    # sem escala: razões e variações relativas
        f["close_sma5"] = c / sma5 - 1.0
        f["close_sma10"] = c / sma10 - 1.0
        f["amplitude"] = (h - l) / c
        f["gap_abertura"] = o / c.shift(1) - 1.0
    f["rsi_14"] = rsi(c, 14)

    nxt = c.shift(-1)
    y = (nxt > c).astype(int).where(nxt.notna())

    ultimo = f.iloc[[-1]].copy() if f.iloc[-1].notna().all() else None
    mask = f.notna().all(axis=1) & y.notna()
    return f[mask], y[mask].astype(int), ultimo


# --------------------------------------------------------------------------
# Divisão, modelos, métricas
# --------------------------------------------------------------------------


def chronological_split(n: int, train_frac: float, val_frac: float):
    i_tr = int(n * train_frac)
    i_va = int(n * (train_frac + val_frac))
    return slice(0, i_tr), slice(i_tr, i_va), slice(i_va, n)


def make_model(name: str, params: dict, seed: int):
    if name == "rf":
        return RandomForestClassifier(**params, random_state=seed, n_jobs=-1)
    if name == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:  # pragma: no cover
            raise ImportError("Instale o XGBoost: pip install xgboost") from e
        return XGBClassifier(**params, random_state=seed, eval_metric="logloss", n_jobs=-1, verbosity=0)
    raise ValueError(name)


def metrics(y_true, y_pred) -> dict:
    return {
        "acuracia": accuracy_score(y_true, y_pred),
        "precisao": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_alta": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "acuracia_balanceada": balanced_accuracy_score(y_true, y_pred),
    }


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Teste de McNemar exato (binomial) sobre acertos pareados."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01 = int(np.sum(a & ~b))   # A acerta, B erra
    n10 = int(np.sum(~a & b))   # A erra, B acerta
    n = n01 + n10
    p = 1.0 if n == 0 else float(binomtest(min(n01, n10), n, 0.5).pvalue)
    return {"a_acerta_b_erra": n01, "a_erra_b_acerta": n10, "p_valor": p}


def tune(name, grid, Xtr, ytr, Xva, yva, seed):
    rows, best, best_score = [], None, -1.0
    for params in ParameterGrid(grid):
        m = make_model(name, params, seed).fit(Xtr, ytr)
        mt = metrics(yva, m.predict(Xva))
        rows.append({**{k: ("None" if v is None else v) for k, v in params.items()}, **mt})
        if mt["f1_macro"] > best_score:
            best, best_score = params, mt["f1_macro"]
    return best, pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Experimento completo
# --------------------------------------------------------------------------


def run_experiment(cfg: Config, df_raw: pd.DataFrame | None = None, progress=None) -> dict:
    say = progress or (lambda msg: None)
    if df_raw is None:
        say("Baixando dados...")
        df_raw = download_data(cfg.ticker, cfg.start, cfg.end, cfg.cache_csv)
    df, rep_limpeza = clean_data(df_raw)
    df = df.loc[cfg.start:cfg.end]
    X, y, X_ultimo = build_features(df, cfg.scenario)
    n = len(X)
    s_tr, s_va, s_te = chronological_split(n, cfg.train_frac, cfg.val_frac)
    Xtr, ytr, Xva, yva, Xte, yte = X.iloc[s_tr], y.iloc[s_tr], X.iloc[s_va], y.iloc[s_va], X.iloc[s_te], y.iloc[s_te]
    Xtv, ytv = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])

    # descritiva
    ret_dia = df["Close"].pct_change().reindex(X.index)
    def _desc(idx, yy):
        return {"inicio": str(idx[0].date()), "fim": str(idx[-1].date()), "n": int(len(idx)),
                "pct_alta": float(yy.mean()), "ret_medio_pct": float(ret_dia.loc[idx].mean() * 100),
                "ret_dp_pct": float(ret_dia.loc[idx].std() * 100)}
    descritiva = {"treino": _desc(Xtr.index, ytr), "validacao": _desc(Xva.index, yva),
                  "teste": _desc(Xte.index, yte), "total": _desc(X.index, y)}

    # ajuste de hiperparâmetros (somente treino -> validação)
    say("Ajustando Random Forest...")
    rf_best, rf_grid_df = tune("rf", cfg.rf_grid, Xtr, ytr, Xva, yva, cfg.seed)
    say("Ajustando XGBoost...")
    xgb_best, xgb_grid_df = tune("xgb", cfg.xgb_grid, Xtr, ytr, Xva, yva, cfg.seed)

    # modelo final: treino+validação -> teste (uma única avaliação)
    say("Treinando modelos finais e avaliando no teste...")
    rf = make_model("rf", rf_best, cfg.seed).fit(Xtv, ytv)
    xgb = make_model("xgb", xgb_best, cfg.seed).fit(Xtv, ytv)
    classe_maj = int(ytv.mode().iloc[0])
    preds = {"Classe majoritária": np.full(len(yte), classe_maj),
             "Random Forest": rf.predict(Xte), "XGBoost": xgb.predict(Xte)}
    yt = yte.to_numpy()
    desempenho = pd.DataFrame({k: metrics(yt, v) for k, v in preds.items()}).T
    conf = {k: confusion_matrix(yt, v, labels=[0, 1]) for k, v in preds.items()}
    correct = {k: (v == yt) for k, v in preds.items()}
    testes = {
        "RF vs XGBoost": mcnemar_exact(correct["Random Forest"], correct["XGBoost"]),
        "RF vs Classe majoritária": mcnemar_exact(correct["Random Forest"], correct["Classe majoritária"]),
        "XGBoost vs Classe majoritária": mcnemar_exact(correct["XGBoost"], correct["Classe majoritária"]),
    }

    # robustez: repete o ajuste final com várias sementes
    say("Rodando verificação de robustez (sementes)...")
    rob = []
    for s in range(cfg.n_seeds_robustez):
        for nome, key, params in (("Random Forest", "rf", rf_best), ("XGBoost", "xgb", xgb_best)):
            p = make_model(key, params, 1000 + s).fit(Xtv, ytv).predict(Xte)
            m = metrics(yt, p)
            rob.append({"modelo": nome, "semente": 1000 + s, "acuracia": m["acuracia"], "f1_macro": m["f1_macro"]})
    rob_df = pd.DataFrame(rob)
    robustez = rob_df.groupby("modelo")[["acuracia", "f1_macro"]].agg(["mean", "std"])

    imp = pd.DataFrame({"Random Forest": rf.feature_importances_, "XGBoost": xgb.feature_importances_}, index=X.columns)

    prox = None
    if X_ultimo is not None:
        prox = {"data_ref": str(X_ultimo.index[0].date()),
                "rf_prob_alta": float(rf.predict_proba(X_ultimo)[0, 1]),
                "xgb_prob_alta": float(xgb.predict_proba(X_ultimo)[0, 1])}

    return {"cfg": cfg, "limpeza": rep_limpeza, "descritiva": descritiva, "features": list(X.columns),
            "rf_best": rf_best, "xgb_best": xgb_best, "rf_grid": rf_grid_df, "xgb_grid": xgb_grid_df,
            "desempenho": desempenho, "confusao": conf, "testes": testes, "robustez": robustez,
            "robustez_raw": rob_df, "importancia": imp, "classe_majoritaria": classe_maj,
            "serie_close": df["Close"], "limites": {"treino_fim": Xtr.index[-1], "val_fim": Xva.index[-1]},
            "proximo_pregao": prox, "modelos": {"rf": rf, "xgb": xgb}}


# --------------------------------------------------------------------------
# Figuras e texto
# --------------------------------------------------------------------------

_COL = {"Classe majoritária": "#8a8a8a", "Random Forest": "#1b7f5c", "XGBoost": "#c4511f"}


def make_figures(res: dict, outdir: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    paths = {}

    # 1) série com divisão
    s = res["serie_close"]
    lim = res["limites"]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(s.index, s.values, color="#333", lw=0.8)
    for a, b, lab, cor in ((s.index[0], lim["treino_fim"], "Treino (60%)", "#1b7f5c"),
                           (lim["treino_fim"], lim["val_fim"], "Validação (20%)", "#d9a400"),
                           (lim["val_fim"], s.index[-1], "Teste (20%)", "#c4511f")):
        ax.axvspan(a, b, color=cor, alpha=0.12, label=lab)
    ax.set_ylabel("Fechamento (pontos)")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    paths["serie"] = os.path.join(outdir, "fig_serie_divisao.png")
    fig.savefig(paths["serie"], dpi=200); plt.close(fig)

    # 2) matrizes de confusão
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3))
    for ax, (nome, cm) in zip(axes, res["confusao"].items()):
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
        ax.set_xticks([0, 1], ["Não alta", "Alta"]); ax.set_yticks([0, 1], ["Não alta", "Alta"])
        ax.set_xlabel("Previsto"); ax.set_title(nome, fontsize=10)
    axes[0].set_ylabel("Real")
    fig.tight_layout()
    paths["confusao"] = os.path.join(outdir, "fig_matrizes_confusao.png")
    fig.savefig(paths["confusao"], dpi=200); plt.close(fig)

    # 3) métricas
    d = res["desempenho"][["acuracia", "f1_alta", "f1_macro"]]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    w = 0.26
    for k, (nome, row) in enumerate(d.iterrows()):
        ax.bar(np.arange(3) + (k - 1) * w, row.values, w, label=nome, color=_COL[nome])
    ax.set_xticks(range(3), ["Acurácia", "F1 (Alta)", "F1 macro"])
    ax.set_ylim(0, 1); ax.legend(frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    paths["metricas"] = os.path.join(outdir, "fig_metricas.png")
    fig.savefig(paths["metricas"], dpi=200); plt.close(fig)

    # 4) importância
    imp = res["importancia"].sort_values("Random Forest")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(imp))
    ax.barh(y - 0.19, imp["Random Forest"], 0.38, label="Random Forest", color=_COL["Random Forest"])
    ax.barh(y + 0.19, imp["XGBoost"], 0.38, label="XGBoost", color=_COL["XGBoost"])
    ax.set_yticks(y, imp.index); ax.set_xlabel("Importância relativa"); ax.legend(frameon=False)
    fig.tight_layout()
    paths["importancia"] = os.path.join(outdir, "fig_importancia.png")
    fig.savefig(paths["importancia"], dpi=200); plt.close(fig)
    return paths


def _f(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def build_results_markdown(res: dict) -> str:
    cfg, D = res["cfg"], res["descritiva"]
    dd, T = res["desempenho"], res["testes"]
    L = []
    L.append(f"# Resultados — cenário {cfg.scenario} ({'preços brutos, projeto original' if cfg.scenario=='A' else 'atributos sem escala'})\n")
    L.append("## Amostra\n")
    L.append("| Conjunto | Início | Fim | N | % Alta | Ret. médio (%) | DP ret. (%) |\n|---|---|---|---|---|---|---|")
    for k in ("treino", "validacao", "teste", "total"):
        v = D[k]
        L.append(f"| {k} | {v['inicio']} | {v['fim']} | {v['n']} | {_f(v['pct_alta']*100,1)} | {_f(v['ret_medio_pct'],3)} | {_f(v['ret_dp_pct'],3)} |")
    L.append(f"\nLimpeza: {res['limpeza']}\n")
    L.append(f"Atributos ({len(res['features'])}): {', '.join(res['features'])}\n")
    L.append(f"Classe majoritária (treino+validação): {'Alta' if res['classe_majoritaria']==1 else 'Não alta'}\n")
    L.append("## Hiperparâmetros escolhidos (maior F1 macro na validação)\n")
    L.append(f"- Random Forest: {res['rf_best']}\n- XGBoost: {res['xgb_best']}\n")
    L.append("## Desempenho no teste\n")
    L.append("| Modelo | Acurácia | Precisão | Recall | F1 (Alta) | F1 macro | Acur. balanc. |\n|---|---|---|---|---|---|---|")
    for nome, r in dd.iterrows():
        L.append(f"| {nome} | {_f(r.acuracia)} | {_f(r.precisao)} | {_f(r.recall)} | {_f(r.f1_alta)} | {_f(r.f1_macro)} | {_f(r.acuracia_balanceada)} |")
    L.append("\n## Teste de McNemar exato (nível de 5%)\n")
    L.append("| Comparação | A acerta/B erra | A erra/B acerta | p-valor | Decisão |\n|---|---|---|---|---|")
    for k, v in T.items():
        dec = "diferença significativa" if v["p_valor"] < 0.05 else "sem diferença significativa"
        L.append(f"| {k} | {v['a_acerta_b_erra']} | {v['a_erra_b_acerta']} | {_f(v['p_valor'],4)} | {dec} |")
    L.append("\n## Robustez (média ± desvio-padrão em %d sementes)\n" % cfg.n_seeds_robustez)
    for nome, r in res["robustez"].iterrows():
        L.append(f"- {nome}: acurácia {_f(r[('acuracia','mean')])} ± {_f(r[('acuracia','std')])}; "
                 f"F1 macro {_f(r[('f1_macro','mean')])} ± {_f(r[('f1_macro','std')])}")

    # frases-base (rascunho automático)
    L.append("\n## Frases-base para o artigo (rascunho automático — revise)\n")
    melhor = dd.loc[["Random Forest", "XGBoost"], "f1_macro"].idxmax()
    cm = dd.loc["Classe majoritária"]
    L.append(f"- Entre os dois algoritmos, o maior F1 macro no teste foi do {melhor} "
             f"({_f(dd.loc[melhor,'f1_macro'])}; F1 da classe Alta {_f(dd.loc[melhor,'f1_alta'])}). "
             f"A classe majoritária obteve F1 macro de {_f(cm.f1_macro)} e F1 da classe Alta de {_f(cm.f1_alta)}.")
    p = T["RF vs XGBoost"]["p_valor"]
    L.append(f"- O teste de McNemar entre Random Forest e XGBoost resultou em p = {_f(p,4)}; "
             f"{'rejeita-se H0' if p < 0.05 else 'não se rejeita H0'} ao nível de 5%.")
    for m, k in (("Random Forest", "RF vs Classe majoritária"), ("XGBoost", "XGBoost vs Classe majoritária")):
        acima = dd.loc[m, "acuracia"] > dd.loc["Classe majoritária", "acuracia"]
        sig = T[k]["p_valor"] < 0.05
        if acima and sig:
            txt = "superou a classe majoritária com significância estatística"
        elif acima:
            txt = "ficou acima da classe majoritária em acurácia, sem significância estatística"
        else:
            txt = "não superou a classe majoritária em acurácia"
        L.append(f"- {m}: {txt} (acurácia {_f(dd.loc[m,'acuracia'])} vs {_f(dd.loc['Classe majoritária','acuracia'])}; p = {_f(T[k]['p_valor'],4)}).")
    return "\n".join(L) + "\n"


def save_all(res: dict, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    res["desempenho"].round(4).to_csv(os.path.join(outdir, "desempenho_teste.csv"), encoding="utf-8-sig")
    res["rf_grid"].round(4).to_csv(os.path.join(outdir, "grade_random_forest.csv"), index=False, encoding="utf-8-sig")
    res["xgb_grid"].round(4).to_csv(os.path.join(outdir, "grade_xgboost.csv"), index=False, encoding="utf-8-sig")
    res["robustez_raw"].to_csv(os.path.join(outdir, "robustez_sementes.csv"), index=False, encoding="utf-8-sig")
    make_figures(res, outdir)
    with open(os.path.join(outdir, "resultados_para_artigo.md"), "w", encoding="utf-8") as fh:
        fh.write(build_results_markdown(res))
