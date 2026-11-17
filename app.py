"""Protótipo Streamlit: coleta, atributos, treino, métricas e matriz de confusão.
Executar: streamlit run app.py
"""
import pandas as pd
import streamlit as st

from ibov_pipeline import Config, build_results_markdown, make_figures, run_experiment

st.set_page_config(page_title="Ibovespa: Random Forest x XGBoost", layout="wide")
st.title("Direção diária do Ibovespa: Random Forest x XGBoost")
st.caption("Ferramenta de pesquisa. Não é recomendação de investimento.")

with st.sidebar:
    st.header("Parâmetros")
    ticker = st.text_input("Símbolo (Yahoo Finance)", "^BVSP")
    ini = st.text_input("Início", "2010-01-01")
    fim = st.text_input("Fim", "2025-12-31")
    cenario = st.radio("Conjunto de atributos", ["B", "A"], format_func=lambda x: {
        "B": "B: atributos sem escala (principal)", "A": "A: preços brutos (projeto original)"}[x])
    sementes = st.slider("Sementes na verificação de robustez", 2, 20, 5)
    rodar = st.button("Executar pipeline", type="primary")


@st.cache_data(show_spinner=False)
def _executar(ticker, ini, fim, cenario, sementes):
    cfg = Config(ticker=ticker, start=ini, end=fim, scenario=cenario, n_seeds_robustez=sementes,
                 cache_csv=None)
    res = run_experiment(cfg)
    res = {k: v for k, v in res.items() if k != "modelos"}   # modelos não são serializáveis pelo cache
    return res


if rodar:
    with st.spinner("Coletando dados e treinando os modelos (pode levar alguns minutos)..."):
        st.session_state["res"] = _executar(ticker, ini, fim, cenario, sementes)

res = st.session_state.get("res")
if res is None:
    st.info("Configure os parâmetros na barra lateral e clique em **Executar pipeline**.")
    st.stop()

D = res["descritiva"]
c1, c2, c3 = st.columns(3)
c1.metric("Observações", D["total"]["n"])
c2.metric("Período de teste", f"{D['teste']['inicio']} a {D['teste']['fim']}")
c3.metric("% de altas (teste)", f"{D['teste']['pct_alta']*100:.1f}%")

aba1, aba2, aba3, aba4 = st.tabs(["Série e divisão", "Desempenho", "Matrizes de confusão", "Testes e detalhes"])

import tempfile
figs = make_figures(res, tempfile.mkdtemp())

with aba1:
    st.image(figs["serie"], caption="Fechamento do índice e divisão cronológica 60/20/20")
    st.dataframe(pd.DataFrame(D).T)
with aba2:
    st.dataframe(res["desempenho"].style.format("{:.3f}"))
    st.image(figs["metricas"])
    st.image(figs["importancia"])
with aba3:
    st.image(figs["confusao"])
with aba4:
    st.subheader("Teste de McNemar exato")
    st.dataframe(pd.DataFrame(res["testes"]).T)
    st.write("Hiperparâmetros escolhidos:", {"Random Forest": res["rf_best"], "XGBoost": res["xgb_best"]})
    st.subheader("Robustez entre sementes")
    st.dataframe(res["robustez"])
    with st.expander("Texto-base dos resultados (Markdown)"):
        st.markdown(build_results_markdown(res))
    if res.get("proximo_pregao"):
        p = res["proximo_pregao"]
        st.subheader("Probabilidade estimada de alta no próximo pregão")
        st.write(f"Dados até {p['data_ref']}: Random Forest {p['rf_prob_alta']:.1%} | XGBoost {p['xgb_prob_alta']:.1%}")
        st.caption("Sem significado econômico: ver as limitações do artigo (custos, slippage, ausência de vantagem estatística).")
