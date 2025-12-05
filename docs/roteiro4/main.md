## 1) Contexto:
Foi treinado um SVM com kernel RBF para classificar clientes em churn (1) e non-churn (0)
a partir da base Telco Customer Churn preprocessada. A avaliação foi realizada em conjunto
de teste com 1.409 registros.

## 2) Resultados numéricos:
Matriz_de_Confusao: "[760 275], [86 288]"

Detalhamento:

TN (0→0): 760

FP (0→1): 275

FN (1→0): 86

TP (1→1): 288

Classification_Report:

Classe_0_nao_churn:

Precision: 0.90
Recall: 0.73
F1_score: 0.81
Support: 1035

Classe_1_churn:

Precision: 0.51
Recall: 0.77
F1_score: 0.61
Support: 374

Accuracy_global:

Valor: 0.74
Total_amostras: 1409

Macro_avg:

Precision: 0.70
Recall: 0.75
F1_score: 0.71

Weighted_avg:

Precision: 0.80
Recall: 0.74
F1_score: 0.76

Metricas_globais:

Acuracia: 0.7438

ROC_AUC: 0.8178

## 3) Interpretação das métricas:
Foco_na_classe_de_negocio_Churn_1:
Recall: 0.77
interpretacao: >
O modelo captura 77% dos clientes que realmente cancelam,
indicando bom desempenho para retenção.

Precision: 0.51
interpretacao: >
Entre os clientes classificados como churn, apenas 51% realmente cancelaram,
indicando alto custo de falso positivo.

F1_score: 0.61
interpretacao: >
Há equilíbrio razoável entre recall e precision, com viés intencional para recall.

Qualidade_global:

ROC_AUC: 0.8178
interpretacao: >
Demonstra boa capacidade de separação entre churners e não churners.

Accuracy: 0.74
interpretacao: >
Adequada, porém não é a principal métrica devido ao leve desbalanceamento da base.

Leitura_da_matriz_de_confusao:
Falsos_positivos: 275
impacto: >
Geração de custo financeiro por ações de retenção mal direcionadas.

Falsos_negativos: 86
impacto: >
Perda potencial de clientes que o modelo não conseguiu identificar.

## 4) Avaliação crítica:
Pontos_fortes:
- Alta taxa de detecção de churners (recall elevado).
- ROC-AUC indica modelo consistente para ranqueamento de risco.
- SVM com kernel RBF captura padrões não lineares relevantes.

Limitacoes:
- Precisão moderada na classe positiva.
- Probabilidades podem exigir calibração.
- Ausência de tuning de hiperparâmetros.

## 5) Recomendações práticas:
Acoes_sugeridas:

- Ajustar limiar de decisão para reduzir falsos positivos.
- Aplicar GridSearchCV para otimizar C e gamma.
- Calibrar probabilidades com CalibratedClassifierCV.
- Realizar análise de custo-benefício entre FP e FN.
- Comparar com modelos baseline como regressão logística e Random Forest.

## 6) Conclusão:

O SVM com kernel RBF apresentou bom poder discriminatório (ROC-AUC ≈ 0.82) e alto recall
para churn, sendo adequado como filtro de risco para ações de retenção. No entanto,
a precisão moderada indica necessidade de tuning e ajuste de limiar para reduzir custos
com falso positivo e maximizar o retorno das campanhas.

=== "Plt"
![SVMplt](SVMplt.svg)


=== "Curva-ROC-SVM"
![Curva-ROC-SVM](Curva-ROC-SVM.png)