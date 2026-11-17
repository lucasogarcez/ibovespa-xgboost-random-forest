# Random Forest vs. XGBoost na direção diária do Ibovespa

Código do artigo **"XGBoost versus Random Forest na previsão da direção diária do Ibovespa: uma análise comparativa com validação temporal estrita"**, desenvolvido na disciplina de Prática de Pesquisa Orientada.

O projeto compara dois algoritmos de árvores de decisão na classificação binária *"o Ibovespa fecha em alta no próximo pregão?"*. A avaliação é estritamente cronológica, sem vazamento de informação futura. Os modelos são comparados entre si e contra uma referência ingênua (prever sempre a classe majoritária), com teste estatístico de McNemar.

> **Aviso:** é um trabalho acadêmico de pesquisa. Nada aqui é recomendação de investimento.

## Principais resultados

Conjunto de teste: 790 pregões, de 31/10/2022 a 29/12/2025 (atributos sem escala, cenário B).

| Modelo | Acurácia | F1 macro | Acurácia balanceada |
|---|---|---|---|
| Classe majoritária (sempre "alta") | 0,511 | 0,338 | 0,500 |
| Random Forest | 0,528 | 0,525 | 0,530 |
| XGBoost | 0,525 | 0,516 | 0,523 |

Nenhuma diferença foi estatisticamente significativa a 5% (McNemar exato): Random Forest vs. XGBoost, p = 0,950; Random Forest vs. majoritária, p = 0,580; XGBoost vs. majoritária, p = 0,562. Nesse desenho, os modelos não superaram uma regra trivial, o que é compatível com a forma fraca da hipótese de eficiência de mercado. O poder estatístico é limitado: com 790 observações, a margem de erro (IC 95%) da acurácia é de cerca de 3,5 pontos percentuais. Detalhes, análise de sensibilidade e limitações estão no artigo.

## Como funciona

1. **Coleta:** cotações diárias de `^BVSP` via [`yfinance`](https://github.com/ranaroussi/yfinance), de 01/01/2010 a 31/12/2025, com cache em CSV.
2. **Limpeza:** ordenação, remoção de duplicatas, ausentes e preços não positivos.
3. **Alvo:** `1` se `Fechamento[t+1] > Fechamento[t]`, senão `0`.
4. **Atributos** (sempre calculados só com dados até o fechamento de `t`):
   - **Cenário B (principal):** retornos e direções dos 3 últimos pregões, distância do fechamento às médias móveis de 5 e 10 dias, amplitude do dia, *gap* de abertura e RSI de 14 dias (11 atributos).
   - **Cenário A (sensibilidade, projeto original):** retornos e direções defasados, níveis de abertura, máxima, mínima e fechamento, médias móveis em pontos e RSI (13 atributos).
5. **Divisão cronológica 60/20/20** (treino, validação, teste), sem embaralhar.
6. **Ajuste de hiperparâmetros** por busca em grade, treinando no treino e escolhendo o maior F1 macro na validação.
7. **Modelo final** reajustado em treino + validação e avaliado **uma única vez** no teste.
8. **Avaliação:** acurácia, precisão, recall, F1 (classe alta), F1 macro, acurácia balanceada, matriz de confusão, McNemar exato e repetição com 10 sementes.

Grades de hiperparâmetros:

| Algoritmo | Hiperparâmetro | Valores |
|---|---|---|
| Random Forest | `n_estimators` | 100, 300 |
| Random Forest | `max_depth` | 3, 5, 8, sem limite |
| XGBoost | `n_estimators` | 100, 300 |
| XGBoost | `max_depth` | 3, 5 |
| XGBoost | `learning_rate` | 0,01, 0,05, 0,1 |

## Estrutura do repositório

```
.
├── ibov_pipeline.py        # núcleo: coleta, atributos, divisão, modelos, métricas, McNemar, figuras
├── run_experimento.py      # roda os cenários B e A e grava tudo em resultados/
├── app.py                  # interface Streamlit (reutiliza ibov_pipeline.py)
├── requirements.txt
├── testes/
│   └── test_pipeline.py    # teste de fumaça com dados sintéticos
├── figuras/
│   ├── gerar_figuras.py    # figuras conceituais do artigo (Random Forest, boosting, divisão)
│   └── *.png
└── resultados/             # gerado ao rodar o experimento
    ├── cenario_B/
    └── cenario_A/
```

## Instalação

Requer Python 3.10 ou superior.

```bash
git clone https://github.com/lucasogarcez/ibovespa-xgboost-random-forest.git
cd ibovespa-xgboost-random-forest
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No Google Colab, basta `!pip install yfinance xgboost` e enviar os arquivos `.py`.

## Uso

### Reproduzir o experimento

```bash
python run_experimento.py
```

O script baixa os dados (salvos em `dados_bvsp.csv`), roda os dois cenários e grava em `resultados/cenario_B/` e `resultados/cenario_A/`:

| Arquivo | Conteúdo |
|---|---|
| `resultados_para_artigo.md` | tabelas e frases-base com os números do artigo |
| `desempenho_teste.csv` | métricas no teste |
| `grade_random_forest.csv`, `grade_xgboost.csv` | resultado de cada combinação de hiperparâmetros na validação |
| `robustez_sementes.csv` | repetição do ajuste final com 10 sementes |
| `fig_*.png` | série com a divisão temporal, matrizes de confusão, métricas e importância dos atributos |

### Interface Streamlit

```bash
streamlit run app.py
```

Na barra lateral, escolha o símbolo, o período e o conjunto de atributos e clique em **Executar pipeline**. O app mostra a série com a divisão temporal, métricas, matrizes de confusão, testes estatísticos e a robustez entre sementes.

### Usar como biblioteca

```python
from ibov_pipeline import Config, run_experiment, save_all

res = run_experiment(Config(scenario="B"))
print(res["desempenho"].round(3))
print(res["testes"])
save_all(res, "resultados/meu_teste")
```

## Reprodutibilidade

- As sementes são fixas (`seed=42` no ajuste principal; 10 sementes extras na verificação de robustez).
- Guarde o `dados_bvsp.csv` gerado: com ele, o experimento reproduz exatamente os mesmos números, sem depender do Yahoo Finance. Os dados são baixados de um serviço de terceiros; confira os termos de uso do Yahoo Finance antes de redistribuí-los.
- O XGBoost, com os parâmetros padrão, não usa sorteio; por isso a semente não altera seus resultados.
- Pequenas diferenças podem aparecer se versões diferentes de `yfinance`, `scikit-learn` ou `xgboost` forem usadas, ou se o Yahoo revisar dados históricos.

## Testes

```bash
python testes/test_pipeline.py
```

O teste usa dados sintéticos e um substituto do XGBoost, e verifica:

- ausência de vazamento (os atributos não mudam quando o futuro distante é alterado);
- correção do alvo;
- ordem cronológica da divisão;
- execução completa dos dois cenários;
- casos conhecidos do McNemar.

Serve apenas para validar o código; os resultados reportados vêm de `run_experimento.py` com os dados reais.

## Limitações

- Uma única divisão cronológica; janelas deslizantes (*walk-forward*) seriam um complemento natural.
- Os regimes de mercado diferem entre treino, validação e teste (a volatilidade no teste é cerca de metade da da validação).
- Atributos apenas de preço e grade de hiperparâmetros restrita, com valores escolhidos na borda da grade.
- A classificação da direção ignora custos de transação, *slippage* e o tamanho dos movimentos.

## Referências principais

- BREIMAN, L. Random forests. *Machine Learning*, v. 45, n. 1, p. 5-32, 2001.
- CHEN, T.; GUESTRIN, C. XGBoost: a scalable tree boosting system. *KDD*, 2016.
- CERQUEIRA, V.; TORGO, L.; MOZETIČ, I. Evaluating time series forecasting models. *Machine Learning*, v. 109, 2020.
- DIETTERICH, T. G. Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, v. 10, n. 7, 1998.
- FAMA, E. F. Efficient capital markets: a review of theory and empirical work. *The Journal of Finance*, v. 25, n. 2, 1970.

A lista completa está no artigo.

## Como citar

```
GARCEZ, Lucas Oliveira. XGBoost versus Random Forest na previsão da direção diária do Ibovespa:
uma análise comparativa com validação temporal estrita. Instituto Federal do Triângulo Mineiro, 2026.
Código disponível em: https://github.com/lucasogarcez/ibovespa-xgboost-random-forest.
```

## Licença

Este projeto está sob a licença MIT. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.

## Autor

Lucas Oliveira Garcez
