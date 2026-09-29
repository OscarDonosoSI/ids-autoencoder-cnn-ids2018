# De qué archivo sale cada resultado del artículo

| Elemento del artículo | Archivo de origen |
|---|---|
| Tabla 1 (distribución de clases) | verificada por `reproducir_pipeline_completo.py`, etapa `limpieza` |
| Tabla 2 (hiperparámetros) | declarada en la Sección 1 del script |
| Tabla 4 (métricas comparativas) | `results/v1.0/autoencoder_resultados.json` y `cnn1d_*_resultados.json` |
| Tabla 5 (matriz binaria) | `results/v1.0/cnn1d_binary_resultados.json` |
| Tabla 6 (clases representativas) | `results/v1.0/cnn1d_multiclass_resultados.json` |
| Tabla 7 (ablación global) | `results/ablacion/RESUMEN_ABLACION.json` |
| Tabla 8 (baselines) | `results/baselines/resultados_rf.json` y `resultados_xgb.json` |
| Tabla 9 (lectura operativa) | derivada de la Tabla 7, cálculo en la Sección 5.2 del artículo |
| Tabla A2 (reporte completo) | `results/v1.0/cnn1d_multiclass_resultados.json` |
| Tabla A3 (ablación por clase) | `results/ablacion/RESUMEN_ABLACION.json` y las seis corridas |
| Figura 1 | `outputs/figures/curvas_entrenamiento_comparacion.png` |
| Figura 2 | `outputs/figures/autoencoder_roc_threshold_analysis.png` |
| Figura 3 | `outputs/figures/cnn1d_multiclass_confusion_matrix.png` |

Las figuras restantes de `outputs/figures/` son del análisis exploratorio y no
aparecen en el artículo.

Los índices de partición (`idx_train/val/test.npy`) no se versionan por tamaño.
Se regeneran de forma determinista con:

    python reproducir_pipeline_completo.py --raiz ./ids2018 --etapa particion
