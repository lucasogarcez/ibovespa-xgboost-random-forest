"""Teste de FUMAÇA com dados sintéticos e um XGBoost de mentira (só para validar o código aqui)."""
import sys, types, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# XGBClassifier falso (HistGradientBoosting) apenas para o teste rodar sem a biblioteca
from sklearn.ensemble import HistGradientBoostingClassifier
class XGBClassifier(HistGradientBoostingClassifier):
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, random_state=0, **_):
        super().__init__(max_iter=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=random_state)
    def get_params(self, deep=True):
        return {"n_estimators": self.max_iter, "max_depth": self.max_depth, "learning_rate": self.learning_rate, "random_state": self.random_state}
    @property
    def feature_importances_(self):
        return np.ones(self.n_features_in_) / self.n_features_in_
sys.modules["xgboost"] = types.SimpleNamespace(XGBClassifier=XGBClassifier)

from ibov_pipeline import *

rng = np.random.default_rng(0)
n = 4000
idx = pd.bdate_range("2010-01-04", periods=n)
c = 70000 * np.exp(np.cumsum(rng.normal(0.0003, 0.013, n)))
o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.003, n))
h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n)))
l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n)))
df = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": 1e6}, index=idx)
df.iloc[10] = np.nan                        # ausente
df = pd.concat([df, df.iloc[[20]]])         # duplicata
df.iloc[50, 3] = 0                          # preço inválido

# 1) ausência de vazamento: o alvo de t depende só de t+1; atributos de t não mudam se mexermos no futuro
d1, _ = clean_data(df)
X1, y1, _ = build_features(d1, "B")
d2 = d1.copy(); d2.iloc[3000:, :4] *= 1.7   # altera o futuro distante
X2, y2, _ = build_features(d2, "B")
common = X1.index[X1.index < d1.index[2990]]
assert np.allclose(X1.loc[common], X2.loc[common]), "VAZAMENTO: atributos mudaram com o futuro"
chk = (d1["Close"].shift(-1) > d1["Close"]).astype(int).reindex(y1.index)
assert (chk == y1).all(), "alvo incorreto"
# 2) divisão cronológica
a, b, c_ = chronological_split(len(X1), .6, .2)
assert X1.index[a][-1] < X1.index[b][0] and X1.index[b][-1] < X1.index[c_][0]
# 3) experimento completo nos dois cenários
import tempfile
for sc in ("A", "B"):
    cfg = Config(scenario=sc, n_seeds_robustez=2, start="2010-01-01", end="2030-12-31", cache_csv=None)
    res = run_experiment(cfg, df_raw=df)
    out = tempfile.mkdtemp(); save_all(res, out)
    print(sc, res["desempenho"].round(3).to_string()); print(res["testes"])
    assert os.path.exists(os.path.join(out, "resultados_para_artigo.md"))
    assert len(res["features"]) == (13 if sc == "A" else 11), res["features"]
# 4) McNemar: casos conhecidos
assert mcnemar_exact(np.ones(10, bool), np.zeros(10, bool))["p_valor"] < 0.01
assert mcnemar_exact(np.ones(5, bool), np.ones(5, bool))["p_valor"] == 1.0
print("OK: todos os testes passaram")
