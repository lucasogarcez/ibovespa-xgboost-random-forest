# Resultados — cenário A (preços brutos, projeto original)

## Amostra

| Conjunto | Início | Fim | N | % Alta | Ret. médio (%) | DP ret. (%) |
|---|---|---|---|---|---|---|
| treino | 2010-01-22 | 2019-08-23 | 2370 | 50,7 | 0,026 | 1,423 |
| validacao | 2019-08-26 | 2022-10-28 | 790 | 52,8 | 0,038 | 1,883 |
| teste | 2022-10-31 | 2025-12-29 | 790 | 51,1 | 0,048 | 0,998 |
| total | 2010-01-22 | 2025-12-29 | 3950 | 51,2 | 0,033 | 1,457 |

Limpeza: {'linhas_brutas': 3965, 'duplicatas_removidas': 0, 'ausentes_removidos': 0, 'precos_nao_positivos_removidos': 0, 'linhas_finais': 3965}

Atributos (13): ret_1, dir_1, ret_2, dir_2, ret_3, dir_3, open, high, low, close, sma_5, sma_10, rsi_14

Classe majoritária (treino+validação): Alta

## Hiperparâmetros escolhidos (maior F1 macro na validação)

- Random Forest: {'max_depth': 3, 'n_estimators': 100}
- XGBoost: {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 300}

## Desempenho no teste

| Modelo | Acurácia | Precisão | Recall | F1 (Alta) | F1 macro | Acur. balanc. |
|---|---|---|---|---|---|---|
| Classe majoritária | 0,511 | 0,511 | 1,000 | 0,677 | 0,338 | 0,500 |
| Random Forest | 0,527 | 0,550 | 0,406 | 0,467 | 0,521 | 0,529 |
| XGBoost | 0,522 | 0,534 | 0,512 | 0,523 | 0,522 | 0,522 |

## Teste de McNemar exato (nível de 5%)

| Comparação | A acerta/B erra | A erra/B acerta | p-valor | Decisão |
|---|---|---|---|---|
| RF vs XGBoost | 129 | 125 | 0,8507 | sem diferença significativa |
| RF vs Classe majoritária | 252 | 240 | 0,6200 | sem diferença significativa |
| XGBoost vs Classe majoritária | 205 | 197 | 0,7270 | sem diferença significativa |

## Robustez (média ± desvio-padrão em 10 sementes)

- Random Forest: acurácia 0,510 ± 0,010; F1 macro 0,500 ± 0,012
- XGBoost: acurácia 0,522 ± 0,000; F1 macro 0,522 ± 0,000

## Conclusões

- Entre os dois algoritmos, o maior F1 macro no teste foi do XGBoost (0,522; F1 da classe Alta 0,523). A classe majoritária obteve F1 macro de 0,338 e F1 da classe Alta de 0,677.
- O teste de McNemar entre Random Forest e XGBoost resultou em p = 0,8507; não se rejeita H0 ao nível de 5%.
- Random Forest: ficou acima da classe majoritária em acurácia, sem significância estatística (acurácia 0,527 vs 0,511; p = 0,6200).
- XGBoost: ficou acima da classe majoritária em acurácia, sem significância estatística (acurácia 0,522 vs 0,511; p = 0,7270).
