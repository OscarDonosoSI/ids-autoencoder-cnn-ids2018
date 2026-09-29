#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reproducir_pipeline_completo.py
================================================================================
Reproduccion integra del estudio

    "Autoencoder versus CNN-1D for Network Traffic Intrusion Detection:
     A Reproducible Comparative Evaluation with Per-Class Analysis on
     CSE-CIC-IDS2018"

    O. E. Donoso, M. A. Pacheco Barreto, J. P. Vargas Infante,
    P. Recalde y R. M. Toasa
    Universidad Tecnologica Israel, Quito, Ecuador
    Journal of Cybersecurity and Privacy - manuscrito jcp-4572356

--------------------------------------------------------------------------------
PROPOSITO
--------------------------------------------------------------------------------
Ejecutar, sin intervencion manual, la cadena completa que va desde la descarga
del conjunto de datos publico hasta la regeneracion de todas las tablas y
figuras del articulo. El script cubre nueve etapas encadenadas:

    1. descarga      Obtener los diez archivos CSV del bucket publico de AWS S3.
    2. limpieza      Depurar y unificar los archivos en formato Parquet.
    3. escalado      Aplicar la transformada logaritmica firmada y el escalado
                     robusto, y materializar la matriz escalada en disco.
    4. particion     Generar la division estratificada 70/15/15 con semilla 42.
    5. autoencoder   Entrenar el autoencoder y calibrar su umbral de decision.
    6. cnn           Entrenar las dos configuraciones de la CNN-1D.
    7. baselines     Entrenar Random Forest y XGBoost sobre la misma particion.
    8. ablacion      Ejecutar las seis corridas de la ablacion de pesos de clase.
    9. reportes      Regenerar las tablas y las figuras del manuscrito.

Cada etapa escribe sus resultados en disco y puede ejecutarse por separado. Al
reanudar, una etapa cuyos productos ya existan se omite, salvo que se indique
--forzar.

--------------------------------------------------------------------------------
USO
--------------------------------------------------------------------------------
    # Ejecutar la cadena completa desde cero
    python reproducir_pipeline_completo.py --raiz ./ids2018 --etapa todo

    # Ejecutar una sola etapa
    python reproducir_pipeline_completo.py --raiz ./ids2018 --etapa limpieza

    # Reejecutar una etapa descartando sus productos previos
    python reproducir_pipeline_completo.py --raiz ./ids2018 --etapa cnn --forzar

    # Consultar el tiempo estimado y el espacio requerido sin ejecutar nada
    python reproducir_pipeline_completo.py --raiz ./ids2018 --etapa todo --simular

--------------------------------------------------------------------------------
REQUISITOS
--------------------------------------------------------------------------------
Espacio en disco : ~85 GB  (6.5 GB crudos, 1.5 GB Parquet, 2.9 GB memmap
                            escalado, el resto temporal durante la limpieza)
Memoria RAM      : 12 GB   (minimo verificado; la matriz escalada no se carga
                            completa en memoria, se accede por memmap)
GPU              : NVIDIA Tesla T4 o superior, 15 GB de VRAM
Tiempo total     : ~9 h en la configuracion de referencia

Las versiones exactas de las dependencias estan fijadas en requirements.txt.

--------------------------------------------------------------------------------
NOTA SOBRE DETERMINISMO
--------------------------------------------------------------------------------
La semilla 42 se fija en Python, NumPy, PyTorch y scikit-learn antes de
cualquier operacion estocastica, lo que hace reproducibles la particion, la
muestra del escalador, la inicializacion de los pesos y el orden de los lotes.
Los nucleos de reduccion de cuDNN no son bit-reproducibles, de modo que dos
ejecuciones identicas sobre GPU pueden diferir en los ultimos decimales. La
magnitud de esa variacion residual se cuantifica en la etapa de ablacion y se
reporta en la Seccion 4.8 del articulo. Para suprimirla por completo es
necesario invocar el script con --determinista, opcion que activa los nucleos
deterministas de cuDNN a costa de aproximadamente un 30 % de tiempo adicional.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ==============================================================================
# SECCION 1. CONSTANTES DEL ESTUDIO
# ==============================================================================
# Los valores de esta seccion definen el experimento publicado. Modificarlos
# produce un experimento distinto, no una reproduccion; por ese motivo se
# declaran aqui de forma explicita y no se exponen como argumentos de linea de
# comandos.

SEMILLA = 42

# Bucket publico del Canadian Institute for Cybersecurity. No requiere
# autenticacion. Los diez archivos corresponden a los diez dias de captura.
S3_BUCKET = "cse-cic-ids2018"
S3_PREFIJO = "Processed Traffic Data for ML Algorithms/"
ARCHIVOS_ORIGEN = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
]

# Tamano de bloque de lectura. Se fija en 200 000 filas para acotar el pico de
# memoria durante la limpieza; el archivo mas grande supera los ocho millones
# de filas y no cabe en los 12 GB de RAM de la plataforma de referencia.
FILAS_POR_BLOQUE = 200_000

# Columnas descartadas: identificadores, campos temporales y los ocho atributos
# de varianza nula detectados durante el analisis exploratorio. Suman catorce.
COLUMNAS_DESCARTADAS = [
    # Identificadores y campos temporales
    "Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp", "Protocol",
    # Atributos de varianza nula
    "Bwd PSH Flags", "Bwd URG Flags", "Fwd Byts/b Avg", "Fwd Pkts/b Avg",
    "Fwd Blk Rate Avg", "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
]

# Columnas en las que persisten valores NaN tras la conversion numerica. El
# origen es el calculo de estadisticos de flujo indefinidos para flujos de un
# solo paquete. Se imputan a cero.
COLUMNAS_CON_NAN = [
    "Active Std", "Fwd PSH Flags", "Fwd URG Flags",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
]

# Valores verificados del conjunto depurado. El script aborta si no se alcanzan,
# de modo que cualquier divergencia respecto del experimento publicado se
# detecta en el momento en que se produce y no se propaga en silencio.
FLUJOS_CRUDOS_ESPERADOS = 13_087_277
FLUJOS_LIMPIOS_ESPERADOS = 11_346_511
N_CARACTERISTICAS = 65
N_CLASES = 15
NAN_IMPUTADOS_ESPERADOS = 3_270_477

# Distribucion de clases del conjunto unificado (Tabla 1 del articulo).
DISTRIBUCION_ESPERADA = {
    "Benign": 9_937_805,
    "DDoS attacks-LOIC-HTTP": 575_772,
    "DDOS attack-HOIC": 276_464,
    "DoS attacks-Hulk": 145_199,
    "Bot": 144_535,
    "Infilteration": 118_468,
    "SSH-Bruteforce": 94_143,
    "DoS attacks-GoldenEye": 41_406,
    "DoS attacks-Slowloris": 9_908,
    "DDOS attack-LOIC-UDP": 1_730,
    "Brute Force-Web": 570,
    "Brute Force-XSS": 229,
    "FTP-BruteForce": 144,
    "SQL Injection": 83,
    "DoS attacks-SlowHTTPTest": 55,
}

# Preprocesamiento
MUESTRA_ESCALADOR = 500_000       # registros de entrenamiento para ajustar el escalador
LIMITE_RECORTE = 10.0             # recorte simetrico posterior al escalado
PROPORCION_ENTRENAMIENTO = 0.70
PROPORCION_VALIDACION = 0.15      # el resto, 0.15, queda para prueba

# Hiperparametros del autoencoder
AE_CAPAS = [128, 64, 16, 64, 128]
AE_DROPOUT = 0.1
AE_EPOCAS_MAX = 30
AE_PACIENCIA = 5
AE_PERCENTILES = np.arange(50.0, 99.6, 0.5)   # barrido de calibracion del umbral
AE_PERCENTIL_REFERENCIA = 95.0

# Hiperparametros de la CNN-1D
CNN_FILTROS = (64, 128)
CNN_KERNEL = 3
CNN_DENSA = 128
CNN_DROPOUT = 0.3
CNN_EPOCAS_MAX = 20
CNN_PACIENCIA = 4

# Comunes a ambos modelos
TAMANO_LOTE = 512
TASA_APRENDIZAJE = 1e-3
PLATEAU_FACTOR = 0.5
PLATEAU_PACIENCIA = 2

# Baselines clasicos. Ambas implementaciones exigen la matriz de entrenamiento
# en memoria, que excede los 12 GB disponibles; por ese motivo se ajustan sobre
# una submuestra estratificada. La asimetria se declara en la Seccion 3.10 del
# articulo y debe conservarse para reproducir la Tabla 8.
BASELINE_SUBMUESTRA = 500_000
RF_ARBOLES = 100
RF_PROFUNDIDAD_MAX = None
XGB_ESTIMADORES = 300
XGB_PROFUNDIDAD_MAX = 8

# Ablacion de pesos de clase
ABLACION_SEMILLAS = (42, 7, 123)
ABLACION_BRAZOS = ("con_pesos", "sin_pesos")

ETAPAS = ["descarga", "limpieza", "escalado", "particion",
          "autoencoder", "cnn", "baselines", "ablacion", "reportes"]


# ==============================================================================
# SECCION 2. INFRAESTRUCTURA: RUTAS, BITACORA Y DETERMINISMO
# ==============================================================================

@dataclass(frozen=True)
class Rutas:
    """Agrupar las rutas derivadas del directorio raiz del experimento."""
    raiz: Path

    @property
    def crudos(self) -> Path: return self.raiz / "data" / "raw"
    @property
    def procesados(self) -> Path: return self.raiz / "data" / "processed"
    @property
    def finales(self) -> Path: return self.raiz / "data" / "final"
    @property
    def modelos(self) -> Path: return self.raiz / "models"
    @property
    def ablacion(self) -> Path: return self.raiz / "models" / "ablacion"
    @property
    def figuras(self) -> Path: return self.raiz / "outputs" / "figures"
    @property
    def tablas(self) -> Path: return self.raiz / "outputs" / "tables"
    @property
    def bitacora(self) -> Path: return self.raiz / "logs"

    def crear(self) -> None:
        """Crear el arbol de directorios completo si aun no existe."""
        for d in (self.crudos, self.procesados, self.finales, self.modelos,
                  self.ablacion, self.figuras, self.tablas, self.bitacora):
            d.mkdir(parents=True, exist_ok=True)


def configurar_bitacora(destino: Path) -> logging.Logger:
    """Instalar una bitacora que escriba de forma simultanea en consola y en disco."""
    destino.mkdir(parents=True, exist_ok=True)
    archivo = destino / f"ejecucion_{time.strftime('%Y%m%d_%H%M%S')}.log"
    formato = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                                datefmt="%H:%M:%S")
    registro = logging.getLogger("ids2018")
    registro.setLevel(logging.INFO)
    registro.handlers.clear()
    for manejador in (logging.StreamHandler(sys.stdout), logging.FileHandler(archivo, encoding="utf-8")):
        manejador.setFormatter(formato)
        registro.addHandler(manejador)
    registro.info("Bitacora de la ejecucion: %s", archivo)
    return registro


def fijar_semilla(semilla: int, determinista: bool = False) -> None:
    """
    Fijar la semilla en todas las fuentes de aleatoriedad del proceso.

    Invocar esta funcion antes de cualquier operacion estocastica. Con
    determinista=True se activan ademas los nucleos deterministas de cuDNN, lo
    que elimina la variabilidad entre ejecuciones a costa de tiempo adicional.
    """
    import torch

    os.environ["PYTHONHASHSEED"] = str(semilla)
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)

    if determinista:
        # CUBLAS exige esta variable para que las reducciones sean reproducibles.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def obtener_dispositivo():
    """Devolver la GPU disponible o, en su defecto, la CPU."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def guardar_json(objeto, destino: Path) -> None:
    """Serializar un objeto a JSON con indentacion, convirtiendo tipos de NumPy."""
    def convertir(x):
        if isinstance(x, (np.integer,)):   return int(x)
        if isinstance(x, (np.floating,)):  return float(x)
        if isinstance(x, np.ndarray):      return x.tolist()
        raise TypeError(f"Tipo no serializable: {type(x)}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(objeto, f, indent=2, ensure_ascii=False, default=convertir)


def verificar(condicion: bool, mensaje: str) -> None:
    """
    Comprobar un invariante del experimento y abortar si no se cumple.

    Usar esta funcion, y no assert, para las verificaciones que deben
    ejecutarse siempre: assert desaparece cuando el interprete corre con -O.
    """
    if not condicion:
        raise RuntimeError(f"Verificacion fallida: {mensaje}")


# ==============================================================================
# SECCION 3. ETAPA 1 - DESCARGA
# ==============================================================================

def etapa_descarga(rutas: Rutas, reg: logging.Logger, forzar: bool) -> None:
    """
    Descargar los diez archivos CSV del bucket publico de AWS S3.

    El bucket admite acceso anonimo, por lo que se configura boto3 sin firma de
    peticion. Un archivo ya presente y con el tamano correcto no se vuelve a
    descargar, salvo que se indique --forzar.
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    reg.info("=" * 78)
    reg.info("ETAPA 1/9 - DESCARGA DE LOS ARCHIVOS DE ORIGEN")
    reg.info("=" * 78)

    cliente = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    for nombre in ARCHIVOS_ORIGEN:
        destino = rutas.crudos / nombre
        clave = S3_PREFIJO + nombre

        # Consultar el tamano remoto para decidir si la descarga es necesaria.
        tamano_remoto = cliente.head_object(Bucket=S3_BUCKET, Key=clave)["ContentLength"]
        if destino.exists() and destino.stat().st_size == tamano_remoto and not forzar:
            reg.info("  [omitido] %s ya presente (%.1f MB)", nombre, tamano_remoto / 1e6)
            continue

        reg.info("  [descarga] %s (%.1f MB)", nombre, tamano_remoto / 1e6)
        cliente.download_file(S3_BUCKET, clave, str(destino))
        verificar(destino.stat().st_size == tamano_remoto,
                  f"la descarga de {nombre} quedo incompleta")

    reg.info("Descarga finalizada: %d archivos en %s", len(ARCHIVOS_ORIGEN), rutas.crudos)


# ==============================================================================
# SECCION 4. ETAPA 2 - LIMPIEZA Y UNIFICACION
# ==============================================================================

def _depurar_bloque(bloque, contadores: Dict[str, int]):
    """
    Depurar un bloque de filas y devolverlo listo para concatenar.

    Las operaciones se aplican en este orden, que no es intercambiable:
      1. eliminar las cabeceras repetidas incrustadas en el cuerpo del CSV,
      2. descartar las catorce columnas sin valor predictivo,
      3. convertir a numerico los campos almacenados como texto,
      4. sustituir los infinitos por NaN,
      5. imputar a cero los NaN residuales de las siete columnas conocidas.

    Invertir 1 y 3 haria que las cabeceras se convirtieran en NaN y se
    imputaran a cero, introduciendo filas espurias en el conjunto.
    """
    import pandas as pd

    # 1. Las cabeceras repetidas se reconocen porque la columna Label contiene
    #    literalmente la cadena "Label".
    antes = len(bloque)
    bloque = bloque[bloque["Label"] != "Label"]
    contadores["cabeceras_eliminadas"] += antes - len(bloque)

    # 2. Descartar identificadores, campos temporales y atributos de varianza nula.
    presentes = [c for c in COLUMNAS_DESCARTADAS if c in bloque.columns]
    bloque = bloque.drop(columns=presentes)

    # 3. Convertir a numerico todas las columnas salvo la etiqueta.
    etiquetas = bloque["Label"]
    rasgos = bloque.drop(columns=["Label"]).apply(pd.to_numeric, errors="coerce")

    # 4. Los infinitos aparecen en Flow Byts/s y Flow Pkts/s cuando la duracion
    #    del flujo es cero. Se convierten a NaN para tratarlos en el paso 5.
    n_infinitos = int(np.isinf(rasgos.to_numpy(dtype="float64", na_value=np.nan)).sum())
    contadores["infinitos_convertidos"] += n_infinitos
    rasgos = rasgos.replace([np.inf, -np.inf], np.nan)

    # 5. Imputar a cero los NaN residuales y contabilizarlos.
    n_nan = int(rasgos.isna().to_numpy().sum())
    contadores["nan_imputados"] += n_nan
    rasgos = rasgos.fillna(0.0)

    rasgos["Label"] = etiquetas.to_numpy()
    return rasgos


def etapa_limpieza(rutas: Rutas, reg: logging.Logger, forzar: bool) -> None:
    """
    Depurar los diez archivos de origen y escribirlos como Parquet.

    Cada archivo se procesa por bloques de 200 000 filas y se guarda de forma
    independiente. La eliminacion de duplicados exactos se realiza al final,
    sobre el conjunto unificado, porque un duplicado puede repartirse entre dos
    archivos distintos.
    """
    import pandas as pd

    reg.info("=" * 78)
    reg.info("ETAPA 2/9 - LIMPIEZA Y UNIFICACION")
    reg.info("=" * 78)

    destino_final = rutas.procesados / "ids2018_limpio.parquet"
    if destino_final.exists() and not forzar:
        reg.info("  [omitido] el conjunto depurado ya existe")
        return

    contadores = {"filas_crudas": 0, "cabeceras_eliminadas": 0,
                  "infinitos_convertidos": 0, "nan_imputados": 0}
    parciales: List[Path] = []

    for indice, nombre in enumerate(ARCHIVOS_ORIGEN, start=1):
        origen = rutas.crudos / nombre
        salida = rutas.procesados / f"{origen.stem}_clean.parquet"
        reg.info("  [%2d/10] %s", indice, nombre)

        trozos = []
        for bloque in pd.read_csv(origen, chunksize=FILAS_POR_BLOQUE, low_memory=False):
            contadores["filas_crudas"] += len(bloque)
            trozos.append(_depurar_bloque(bloque, contadores))

        depurado = pd.concat(trozos, ignore_index=True)
        depurado.to_parquet(salida, index=False, compression="snappy")
        parciales.append(salida)
        reg.info("          %s filas depuradas", f"{len(depurado):,}")
        del trozos, depurado

    # Unificar los diez archivos y eliminar duplicados exactos sobre el total.
    reg.info("  Unificando y eliminando duplicados exactos ...")
    unificado = pd.concat([pd.read_parquet(p) for p in parciales], ignore_index=True)
    antes = len(unificado)
    unificado = unificado.drop_duplicates(ignore_index=True)
    contadores["duplicados_eliminados"] = antes - len(unificado)

    # Verificar los invariantes del conjunto publicado antes de continuar.
    verificar(contadores["filas_crudas"] == FLUJOS_CRUDOS_ESPERADOS,
              f"se leyeron {contadores['filas_crudas']:,} filas crudas, "
              f"se esperaban {FLUJOS_CRUDOS_ESPERADOS:,}")
    verificar(len(unificado) == FLUJOS_LIMPIOS_ESPERADOS,
              f"el conjunto depurado tiene {len(unificado):,} flujos, "
              f"se esperaban {FLUJOS_LIMPIOS_ESPERADOS:,}")
    verificar(unificado.shape[1] - 1 == N_CARACTERISTICAS,
              f"se obtuvieron {unificado.shape[1] - 1} caracteristicas, "
              f"se esperaban {N_CARACTERISTICAS}")
    verificar(contadores["nan_imputados"] == NAN_IMPUTADOS_ESPERADOS,
              f"se imputaron {contadores['nan_imputados']:,} valores NaN, "
              f"se esperaban {NAN_IMPUTADOS_ESPERADOS:,}")

    # Contrastar la distribucion de clases con la Tabla 1 del articulo.
    observada = unificado["Label"].value_counts().to_dict()
    verificar(len(observada) == N_CLASES,
              f"se hallaron {len(observada)} clases, se esperaban {N_CLASES}")
    for clase, esperado in DISTRIBUCION_ESPERADA.items():
        verificar(observada.get(clase, 0) == esperado,
                  f"la clase {clase} tiene {observada.get(clase, 0):,} instancias, "
                  f"se esperaban {esperado:,}")

    unificado.to_parquet(destino_final, index=False, compression="snappy")
    guardar_json(contadores, rutas.tablas / "conteos_limpieza.json")

    reg.info("  Filas crudas leidas ........ %s", f"{contadores['filas_crudas']:,}")
    reg.info("  Cabeceras eliminadas ....... %s", f"{contadores['cabeceras_eliminadas']:,}")
    reg.info("  Infinitos convertidos ...... %s", f"{contadores['infinitos_convertidos']:,}")
    reg.info("  Valores NaN imputados ...... %s", f"{contadores['nan_imputados']:,}")
    reg.info("  Duplicados eliminados ...... %s", f"{contadores['duplicados_eliminados']:,}")
    reg.info("  Conjunto final ............. %s flujos x %d caracteristicas",
             f"{len(unificado):,}", N_CARACTERISTICAS)


# ==============================================================================
# SECCION 5. ETAPA 3 - TRANSFORMACION Y ESCALADO
# ==============================================================================

def _identificar_colas_pesadas(matriz: np.ndarray, umbral_curtosis: float = 10.0) -> np.ndarray:
    """
    Identificar las variables de cola pesada que requieren la transformada
    logaritmica firmada.

    El criterio es la curtosis de Fisher. Con el conjunto publicado el criterio
    selecciona 54 de las 65 variables; el script verifica ese numero, ya que un
    valor distinto indicaria que la entrada no corresponde al experimento.
    """
    from scipy.stats import kurtosis
    curtosis = kurtosis(matriz, axis=0, fisher=True, nan_policy="omit")
    return np.where(np.abs(curtosis) > umbral_curtosis)[0]


def etapa_escalado(rutas: Rutas, reg: logging.Logger, forzar: bool) -> None:
    """
    Aplicar la transformada logaritmica firmada y el escalado robusto, y
    materializar la matriz resultante como memmap de float32.

    La transformada sign(x)*log(1+|x|) es indispensable: sin ella el
    autoencoder produce perdidas del orden de 10^15 y no converge. El escalador
    se ajusta sobre una muestra de 500 000 registros de entrenamiento
    exclusivamente; ni validacion ni prueba intervienen en el ajuste.
    """
    import pandas as pd
    import joblib
    from sklearn.preprocessing import RobustScaler, LabelEncoder

    reg.info("=" * 78)
    reg.info("ETAPA 3/9 - TRANSFORMACION Y ESCALADO")
    reg.info("=" * 78)

    ruta_memmap = rutas.finales / "X_escalado.memmap"
    if ruta_memmap.exists() and not forzar:
        reg.info("  [omitido] la matriz escalada ya existe")
        return

    fijar_semilla(SEMILLA)
    datos = pd.read_parquet(rutas.procesados / "ids2018_limpio.parquet")
    etiquetas_texto = datos["Label"].to_numpy()
    X = datos.drop(columns=["Label"]).to_numpy(dtype=np.float64)
    del datos

    # Codificar las etiquetas. El orden alfabetico que impone LabelEncoder es
    # el que se usa en todas las matrices de confusion del articulo.
    codificador = LabelEncoder().fit(etiquetas_texto)
    y_multi = codificador.transform(etiquetas_texto).astype(np.int64)
    indice_benigno = int(np.where(codificador.classes_ == "Benign")[0][0])
    y_binario = (y_multi != indice_benigno).astype(np.int64)

    # Seleccionar las variables de cola pesada y aplicar la transformada.
    idx_log = _identificar_colas_pesadas(X)
    verificar(len(idx_log) == 54,
              f"se identificaron {len(idx_log)} variables de cola pesada, se esperaban 54")
    reg.info("  Aplicando la transformada logaritmica firmada a %d variables", len(idx_log))
    X[:, idx_log] = np.sign(X[:, idx_log]) * np.log1p(np.abs(X[:, idx_log]))

    # Reservar la particion antes de ajustar el escalador, de modo que la
    # muestra provenga unicamente del conjunto de entrenamiento.
    idx_ent, idx_val, idx_pru = _construir_particion(y_multi)
    generador = np.random.default_rng(SEMILLA)
    muestra = generador.choice(idx_ent, size=MUESTRA_ESCALADOR, replace=False)

    reg.info("  Ajustando RobustScaler sobre %s registros de entrenamiento",
             f"{MUESTRA_ESCALADOR:,}")
    escalador = RobustScaler().fit(X[muestra])

    # Materializar la matriz escalada en disco por bloques, para no duplicar en
    # memoria una matriz de 11.3 millones de filas.
    reg.info("  Escribiendo la matriz escalada en %s", ruta_memmap.name)
    salida = np.memmap(ruta_memmap, dtype=np.float32, mode="w+",
                       shape=(X.shape[0], N_CARACTERISTICAS))
    for inicio in range(0, X.shape[0], FILAS_POR_BLOQUE):
        fin = min(inicio + FILAS_POR_BLOQUE, X.shape[0])
        bloque = escalador.transform(X[inicio:fin])
        # Recortar a (-10, 10) como salvaguarda frente a atipicos residuales.
        np.clip(bloque, -LIMITE_RECORTE, LIMITE_RECORTE, out=bloque)
        salida[inicio:fin] = bloque.astype(np.float32)
    salida.flush()
    del salida, X

    # Persistir los artefactos necesarios para reaplicar la transformacion.
    joblib.dump(escalador, rutas.finales / "scaler.pkl")
    joblib.dump(codificador, rutas.finales / "label_encoder.pkl")
    joblib.dump(idx_log, rutas.finales / "log_features_idx.pkl")
    np.save(rutas.finales / "y_multi.npy", y_multi)
    np.save(rutas.finales / "y_binary.npy", y_binario)
    guardar_json(
        {
            "n_flujos": int(y_multi.shape[0]),
            "n_caracteristicas": N_CARACTERISTICAS,
            "n_clases": N_CLASES,
            "variables_transformadas": len(idx_log),
            "muestra_escalador": MUESTRA_ESCALADOR,
            "limite_recorte": LIMITE_RECORTE,
            "semilla": SEMILLA,
            "clases": list(codificador.classes_),
        },
        rutas.finales / "metadata.json",
    )
    reg.info("  Escalado finalizado")


# ==============================================================================
# SECCION 6. ETAPA 4 - PARTICION ESTRATIFICADA
# ==============================================================================

def _construir_particion(y_multi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construir la division 70/15/15 estratificada sobre la etiqueta de 15 clases.

    Se aplica train_test_split dos veces: la primera separa el 30 % que reunira
    validacion y prueba, y la segunda lo divide por la mitad. Estratificar en
    ambos pasos mantiene la proporcion de cada clase en los tres subconjuntos
    con una desviacion inferior a 0.01 puntos porcentuales.
    """
    from sklearn.model_selection import train_test_split

    indices = np.arange(y_multi.shape[0])
    idx_ent, idx_resto = train_test_split(
        indices, train_size=PROPORCION_ENTRENAMIENTO,
        stratify=y_multi, random_state=SEMILLA)
    idx_val, idx_pru = train_test_split(
        idx_resto, train_size=0.5,
        stratify=y_multi[idx_resto], random_state=SEMILLA)
    return np.sort(idx_ent), np.sort(idx_val), np.sort(idx_pru)


def etapa_particion(rutas: Rutas, reg: logging.Logger, forzar: bool) -> None:
    """Generar y persistir los indices de las tres particiones."""
    reg.info("=" * 78)
    reg.info("ETAPA 4/9 - PARTICION ESTRATIFICADA 70/15/15")
    reg.info("=" * 78)

    if (rutas.finales / "idx_test.npy").exists() and not forzar:
        reg.info("  [omitido] los indices de particion ya existen")
        return

    fijar_semilla(SEMILLA)
    y_multi = np.load(rutas.finales / "y_multi.npy")
    idx_ent, idx_val, idx_pru = _construir_particion(y_multi)

    verificar(len(idx_ent) == 7_947_096, f"entrenamiento: {len(idx_ent):,} flujos")
    verificar(len(idx_val) == 1_697_438, f"validacion: {len(idx_val):,} flujos")
    verificar(len(idx_pru) == 1_701_977, f"prueba: {len(idx_pru):,} flujos")

    np.save(rutas.finales / "idx_train.npy", idx_ent)
    np.save(rutas.finales / "idx_val.npy", idx_val)
    np.save(rutas.finales / "idx_test.npy", idx_pru)

    for nombre, idx in (("entrenamiento", idx_ent), ("validacion", idx_val), ("prueba", idx_pru)):
        reg.info("  %-14s %s flujos", nombre, f"{len(idx):,}")


# ==============================================================================
# SECCION 7. CARGA EFICIENTE DE DATOS
# ==============================================================================

class CargadorRapido:
    """
    Iterar sobre lotes leidos directamente de un memmap de float32.

    Se prescinde de torch.utils.data.DataLoader porque su indexacion elemento a
    elemento sobre un memmap de 11.3 millones de filas domina el tiempo de
    entrenamiento. Leer rebanadas contiguas y transferirlas en bloque a la GPU
    reduce el tiempo por epoca en torno a un orden de magnitud.
    """

    def __init__(self, memmap, indices: np.ndarray, etiquetas: np.ndarray,
                 tamano_lote: int, dispositivo, mezclar: bool = False,
                 generador: np.random.Generator | None = None):
        self.memmap = memmap
        self.indices = indices
        self.etiquetas = etiquetas
        self.tamano_lote = tamano_lote
        self.dispositivo = dispositivo
        self.mezclar = mezclar
        self.generador = generador or np.random.default_rng(SEMILLA)

    def __len__(self) -> int:
        return (len(self.indices) + self.tamano_lote - 1) // self.tamano_lote

    def __iter__(self):
        import torch

        orden = self.indices
        if self.mezclar:
            orden = orden[self.generador.permutation(len(orden))]

        for inicio in range(0, len(orden), self.tamano_lote):
            seleccion = np.sort(orden[inicio:inicio + self.tamano_lote])
            X = torch.from_numpy(np.asarray(self.memmap[seleccion])).to(
                self.dispositivo, non_blocking=True)
            y = torch.from_numpy(self.etiquetas[seleccion]).to(
                self.dispositivo, non_blocking=True)
            yield X, y


def abrir_memmap(rutas: Rutas) -> np.memmap:
    """Abrir en modo solo lectura la matriz escalada materializada en disco."""
    n = np.load(rutas.finales / "y_multi.npy").shape[0]
    return np.memmap(rutas.finales / "X_escalado.memmap", dtype=np.float32,
                     mode="r", shape=(n, N_CARACTERISTICAS))


# ==============================================================================
# SECCION 8. DEFINICION DE LAS ARQUITECTURAS
# ==============================================================================

def construir_autoencoder(n_entrada: int = N_CARACTERISTICAS):
    """
    Construir el autoencoder simetrico 65-128-64-16-64-128-65.

    El cuello de botella de 16 dimensiones obliga al codificador a comprimir el
    trafico benigno en una representacion compacta. En inferencia el error de
    reconstruccion actua como puntuacion de anomalia.
    """
    import torch.nn as nn

    capas: List[nn.Module] = []
    dimensiones = [n_entrada] + AE_CAPAS + [n_entrada]
    for i in range(len(dimensiones) - 1):
        capas.append(nn.Linear(dimensiones[i], dimensiones[i + 1]))
        # La capa de salida se deja lineal: reconstruye valores escalados con signo.
        if i < len(dimensiones) - 2:
            capas.append(nn.ReLU())
            capas.append(nn.Dropout(AE_DROPOUT))
    return nn.Sequential(*capas)


def construir_cnn1d(n_clases: int, n_entrada: int = N_CARACTERISTICAS):
    """
    Construir la CNN-1D que trata el vector de 65 caracteristicas como una
    secuencia de un solo canal.

    Dos bloques convolucionales, el segundo con submuestreo por maximo de factor
    dos, producen una representacion aplanada de 4096 unidades que alimenta una
    capa densa de 128 con dropout 0.3 antes de la salida.
    """
    import torch
    import torch.nn as nn

    class CNN1D(nn.Module):
        def __init__(self):
            super().__init__()
            f1, f2 = CNN_FILTROS
            relleno = CNN_KERNEL // 2
            self.extractor = nn.Sequential(
                nn.Conv1d(1, f1, kernel_size=CNN_KERNEL, padding=relleno),
                nn.BatchNorm1d(f1), nn.ReLU(),
                nn.Conv1d(f1, f2, kernel_size=CNN_KERNEL, padding=relleno),
                nn.BatchNorm1d(f2), nn.ReLU(),
                nn.MaxPool1d(2),
            )
            self.clasificador = nn.Sequential(
                nn.Flatten(),
                nn.Linear(f2 * (n_entrada // 2), CNN_DENSA), nn.ReLU(),
                nn.Dropout(CNN_DROPOUT),
                nn.Linear(CNN_DENSA, n_clases),
            )

        def forward(self, x):
            return self.clasificador(self.extractor(x.unsqueeze(1)))

    return CNN1D()


def contar_parametros(modelo) -> int:
    """Contabilizar los parametros entrenables de un modelo."""
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


# ==============================================================================
# SECCION 9. ETAPA 5 - AUTOENCODER
# ==============================================================================

def etapa_autoencoder(rutas: Rutas, reg: logging.Logger, forzar: bool,
                      determinista: bool) -> None:
    """
    Entrenar el autoencoder sobre trafico benigno y calibrar su umbral.

    El entrenamiento emplea unicamente los flujos benignos de la particion de
    entrenamiento, de modo que el modelo aprende una nocion de normalidad sin
    haber visto ningun ataque. El umbral se elige despues barriendo los
    percentiles 50 a 99.5 del error de reconstruccion sobre los flujos benignos
    de validacion y quedandose con el que maximiza el F1 del conjunto de
    validacion completo. Ni la particion de prueba ni sus etiquetas intervienen.
    """
    import torch
    import torch.nn as nn
    from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score)

    reg.info("=" * 78)
    reg.info("ETAPA 5/9 - AUTOENCODER")
    reg.info("=" * 78)

    destino = rutas.modelos / "autoencoder_resultados.json"
    if destino.exists() and not forzar:
        reg.info("  [omitido] los resultados del autoencoder ya existen")
        return

    fijar_semilla(SEMILLA, determinista)
    dispositivo = obtener_dispositivo()
    X = abrir_memmap(rutas)
    y_bin = np.load(rutas.finales / "y_binary.npy")
    idx_ent = np.load(rutas.finales / "idx_train.npy")
    idx_val = np.load(rutas.finales / "idx_val.npy")
    idx_pru = np.load(rutas.finales / "idx_test.npy")

    # Restringir el entrenamiento a los flujos benignos.
    idx_ent_benigno = idx_ent[y_bin[idx_ent] == 0]
    idx_val_benigno = idx_val[y_bin[idx_val] == 0]
    verificar(len(idx_ent_benigno) == 6_960_438,
              f"trafico benigno de entrenamiento: {len(idx_ent_benigno):,} flujos")

    modelo = construir_autoencoder().to(dispositivo)
    reg.info("  Parametros entrenables: %s", f"{contar_parametros(modelo):,}")

    criterio = nn.MSELoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=TASA_APRENDIZAJE)
    planificador = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizador, mode="min", factor=PLATEAU_FACTOR, patience=PLATEAU_PACIENCIA)

    cargador_ent = CargadorRapido(X, idx_ent_benigno, y_bin, TAMANO_LOTE,
                                  dispositivo, mezclar=True)
    cargador_val = CargadorRapido(X, idx_val_benigno, y_bin, TAMANO_LOTE, dispositivo)

    mejor_perdida, sin_mejora, historial = float("inf"), 0, []
    for epoca in range(1, AE_EPOCAS_MAX + 1):
        modelo.train()
        acumulada, lotes = 0.0, 0
        for xb, _ in cargador_ent:
            optimizador.zero_grad(set_to_none=True)
            perdida = criterio(modelo(xb), xb)
            perdida.backward()
            optimizador.step()
            acumulada += perdida.item(); lotes += 1
        perdida_ent = acumulada / lotes

        modelo.eval()
        acumulada, lotes = 0.0, 0
        with torch.no_grad():
            for xb, _ in cargador_val:
                acumulada += criterio(modelo(xb), xb).item(); lotes += 1
        perdida_val = acumulada / lotes

        planificador.step(perdida_val)
        historial.append({"epoca": epoca, "perdida_entrenamiento": perdida_ent,
                          "perdida_validacion": perdida_val})
        reg.info("  epoca %2d/%d  perdida_ent=%.6f  perdida_val=%.6f",
                 epoca, AE_EPOCAS_MAX, perdida_ent, perdida_val)

        # Detencion temprana sobre la perdida de validacion.
        if perdida_val < mejor_perdida:
            mejor_perdida, sin_mejora = perdida_val, 0
            torch.save(modelo.state_dict(), rutas.modelos / "autoencoder_best.pt")
        else:
            sin_mejora += 1
            if sin_mejora >= AE_PACIENCIA:
                reg.info("  Detencion temprana en la epoca %d", epoca)
                break

    modelo.load_state_dict(torch.load(rutas.modelos / "autoencoder_best.pt"))
    modelo.eval()

    def errores_reconstruccion(indices: np.ndarray) -> np.ndarray:
        """Calcular el error cuadratico medio por flujo para un conjunto de indices."""
        salida = np.empty(len(indices), dtype=np.float64)
        posicion = 0
        with torch.no_grad():
            for xb, _ in CargadorRapido(X, indices, np.zeros(len(X), dtype=np.int64),
                                        TAMANO_LOTE, dispositivo):
                e = ((modelo(xb) - xb) ** 2).mean(dim=1).cpu().numpy()
                salida[posicion:posicion + len(e)] = e
                posicion += len(e)
        return salida

    # Calibrar el umbral sobre validacion.
    reg.info("  Calibrando el umbral sobre la particion de validacion")
    err_val = errores_reconstruccion(idx_val)
    err_val_benigno = err_val[y_bin[idx_val] == 0]
    barrido = []
    for percentil in AE_PERCENTILES:
        umbral = float(np.percentile(err_val_benigno, percentil))
        pred = (err_val > umbral).astype(np.int64)
        p, r, f1, _ = precision_recall_fscore_support(
            y_bin[idx_val], pred, average="binary", zero_division=0)
        barrido.append({"percentil": float(percentil), "umbral": umbral,
                        "precision": p, "recall": r, "f1": f1})

    mejor = max(barrido, key=lambda d: d["f1"])
    referencia = min(barrido, key=lambda d: abs(d["percentil"] - AE_PERCENTIL_REFERENCIA))
    reg.info("  Percentil optimo: %.1f (umbral MSE = %.6f)",
             mejor["percentil"], mejor["umbral"])

    # Evaluar ambos puntos de operacion sobre la particion de prueba.
    err_pru = errores_reconstruccion(idx_pru)
    y_pru = y_bin[idx_pru]
    auc = float(roc_auc_score(y_pru, err_pru))

    resultados = {"auc_roc": auc, "historial": historial, "barrido_umbral": barrido,
                  "puntos_operacion": {}}
    for etiqueta, punto in (("p95_referencia", referencia), ("optimo", mejor)):
        pred = (err_pru > punto["umbral"]).astype(np.int64)
        p, r, f1, _ = precision_recall_fscore_support(
            y_pru, pred, average="binary", zero_division=0)
        falsos_positivos = int(((pred == 1) & (y_pru == 0)).sum())
        resultados["puntos_operacion"][etiqueta] = {
            "percentil": punto["percentil"], "umbral_mse": punto["umbral"],
            "precision": float(p), "recall": float(r), "f1": float(f1),
            "auc_roc": auc,
            "falsos_positivos": falsos_positivos,
            "tasa_falsos_positivos": falsos_positivos / int((y_pru == 0).sum()),
        }
        reg.info("  %-16s P=%.4f  R=%.4f  F1=%.4f  TFP=%.4f", etiqueta, p, r, f1,
                 resultados["puntos_operacion"][etiqueta]["tasa_falsos_positivos"])

    guardar_json(resultados, destino)


# ==============================================================================
# SECCION 10. ENTRENAMIENTO SUPERVISADO (NUCLEO COMPARTIDO)
# ==============================================================================

def calcular_pesos_clase(y_entrenamiento: np.ndarray, n_clases: int) -> np.ndarray:
    """
    Calcular los pesos balanceados inversamente proporcionales a la frecuencia.

    Corresponde al esquema 'balanced' de scikit-learn. Con el conjunto publicado
    los pesos abarcan mas de cinco ordenes de magnitud, de 0.08 para la clase
    Benign a 13 584.78 para DoS attacks-SlowHTTPTest.
    """
    from sklearn.utils.class_weight import compute_class_weight
    return compute_class_weight("balanced",
                                classes=np.arange(n_clases),
                                y=y_entrenamiento).astype(np.float32)


def entrenar_cnn(X, y, idx_ent, idx_val, n_clases, semilla, usar_pesos,
                 dispositivo, reg, etiqueta):
    """
    Entrenar una instancia de la CNN-1D y devolver el mejor modelo y su historial.

    Esta funcion es el nucleo compartido por la etapa 6 y por la ablacion de la
    etapa 8. El unico parametro que distingue los dos brazos de la ablacion es
    usar_pesos; todo lo demas, arquitectura, optimizador, planificador, tamano
    de lote, criterio de parada, particiones y orden de los datos, permanece
    identico.
    """
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score

    fijar_semilla(semilla)
    modelo = construir_cnn1d(n_clases).to(dispositivo)

    if usar_pesos:
        pesos = torch.from_numpy(calcular_pesos_clase(y[idx_ent], n_clases)).to(dispositivo)
        criterio = nn.CrossEntropyLoss(weight=pesos)
    else:
        criterio = nn.CrossEntropyLoss()

    optimizador = torch.optim.Adam(modelo.parameters(), lr=TASA_APRENDIZAJE)
    planificador = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizador, mode="max", factor=PLATEAU_FACTOR, patience=PLATEAU_PACIENCIA)

    generador = np.random.default_rng(semilla)
    cargador_ent = CargadorRapido(X, idx_ent, y, TAMANO_LOTE, dispositivo,
                                  mezclar=True, generador=generador)
    cargador_val = CargadorRapido(X, idx_val, y, TAMANO_LOTE, dispositivo)

    mejor_f1, sin_mejora, historial, mejores_pesos = -1.0, 0, [], None
    for epoca in range(1, CNN_EPOCAS_MAX + 1):
        modelo.train()
        acumulada, lotes = 0.0, 0
        for xb, yb in cargador_ent:
            optimizador.zero_grad(set_to_none=True)
            perdida = criterio(modelo(xb), yb)
            perdida.backward()
            optimizador.step()
            acumulada += perdida.item(); lotes += 1
        perdida_ent = acumulada / lotes

        modelo.eval()
        predichas, reales = [], []
        with torch.no_grad():
            for xb, yb in cargador_val:
                predichas.append(modelo(xb).argmax(dim=1).cpu().numpy())
                reales.append(yb.cpu().numpy())
        f1_macro = float(f1_score(np.concatenate(reales), np.concatenate(predichas),
                                  average="macro", zero_division=0))

        planificador.step(f1_macro)
        historial.append({"epoca": epoca, "perdida_entrenamiento": perdida_ent,
                          "f1_macro_validacion": f1_macro})
        reg.info("  [%s] epoca %2d/%d  perdida=%.4f  F1-macro_val=%.4f",
                 etiqueta, epoca, CNN_EPOCAS_MAX, perdida_ent, f1_macro)

        # Detencion temprana sobre el F1-macro de validacion. Se prefiere esta
        # magnitud a la perdida porque el desbalance severo hace de la perdida
        # un criterio de parada poco fiable.
        if f1_macro > mejor_f1:
            mejor_f1, sin_mejora = f1_macro, 0
            mejores_pesos = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= CNN_PACIENCIA:
                reg.info("  [%s] detencion temprana en la epoca %d", etiqueta, epoca)
                break

    modelo.load_state_dict(mejores_pesos)
    return modelo, historial


def evaluar_cnn(modelo, X, y, idx_pru, n_clases, dispositivo, nombres_clase=None):
    """Evaluar un modelo entrenado sobre la particion de prueba y devolver las metricas."""
    import torch
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 roc_auc_score, average_precision_score)

    modelo.eval()
    predichas, reales, probabilidades = [], [], []
    with torch.no_grad():
        for xb, yb in CargadorRapido(X, idx_pru, y, TAMANO_LOTE, dispositivo):
            logits = modelo(xb)
            predichas.append(logits.argmax(dim=1).cpu().numpy())
            reales.append(yb.cpu().numpy())
            if n_clases == 2:
                probabilidades.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

    y_pred = np.concatenate(predichas)
    y_real = np.concatenate(reales)
    etiquetas = nombres_clase if nombres_clase else [str(i) for i in range(n_clases)]

    resultado = {
        "reporte": classification_report(y_real, y_pred, target_names=etiquetas,
                                         output_dict=True, zero_division=0, digits=4),
        "matriz_confusion": confusion_matrix(y_real, y_pred).tolist(),
    }
    if n_clases == 2:
        puntajes = np.concatenate(probabilidades)
        resultado["auc_roc"] = float(roc_auc_score(y_real, puntajes))
        resultado["auc_pr"] = float(average_precision_score(y_real, puntajes))
    return resultado


def etapa_cnn(rutas: Rutas, reg: logging.Logger, forzar: bool, determinista: bool) -> None:
    """Entrenar y evaluar las dos configuraciones de la CNN-1D, binaria y de 15 clases."""
    import torch
    import joblib

    reg.info("=" * 78)
    reg.info("ETAPA 6/9 - CNN-1D BINARIA Y MULTICLASE")
    reg.info("=" * 78)

    fijar_semilla(SEMILLA, determinista)
    dispositivo = obtener_dispositivo()
    X = abrir_memmap(rutas)
    idx_ent = np.load(rutas.finales / "idx_train.npy")
    idx_val = np.load(rutas.finales / "idx_val.npy")
    idx_pru = np.load(rutas.finales / "idx_test.npy")
    clases = list(joblib.load(rutas.finales / "label_encoder.pkl").classes_)

    configuraciones = [
        ("binary", np.load(rutas.finales / "y_binary.npy"), 2, ["Benign", "Attack"]),
        ("multiclass", np.load(rutas.finales / "y_multi.npy"), N_CLASES, clases),
    ]

    for nombre, y, n_clases, etiquetas in configuraciones:
        destino = rutas.modelos / f"cnn1d_{nombre}_resultados.json"
        if destino.exists() and not forzar:
            reg.info("  [omitido] los resultados de la CNN-1D %s ya existen", nombre)
            continue

        reg.info("  Entrenando la configuracion %s (%d clases)", nombre, n_clases)
        modelo, historial = entrenar_cnn(X, y, idx_ent, idx_val, n_clases,
                                         SEMILLA, True, dispositivo, reg, nombre)
        reg.info("  Parametros entrenables: %s", f"{contar_parametros(modelo):,}")

        resultado = evaluar_cnn(modelo, X, y, idx_pru, n_clases, dispositivo, etiquetas)
        resultado["historial"] = historial
        resultado["parametros"] = contar_parametros(modelo)
        resultado["semilla"] = SEMILLA

        torch.save(modelo.state_dict(), rutas.modelos / f"cnn1d_{nombre}_best.pt")
        guardar_json(resultado, destino)

        promedio = "weighted avg" if n_clases == 2 else "macro avg"
        m = resultado["reporte"][promedio]
        reg.info("  %s -> P=%.4f  R=%.4f  F1=%.4f", nombre,
                 m["precision"], m["recall"], m["f1-score"])


# ==============================================================================
# SECCION 11. ETAPA 7 - BASELINES CLASICOS
# ==============================================================================

def etapa_baselines(rutas: Rutas, reg: logging.Logger, forzar: bool) -> None:
    """
    Entrenar Random Forest y XGBoost sobre la misma particion de prueba.

    Ambas implementaciones exigen la matriz de entrenamiento completa en
    memoria, lo que excede los 12 GB de la plataforma de referencia. Por ese
    motivo se ajustan sobre una submuestra estratificada de 500 000 flujos que
    conserva las proporciones de clase del conjunto de entrenamiento. La
    particion de prueba es identica a la de los modelos profundos, de modo que
    la comparacion de la Tabla 8 se realiza sobre los mismos 1 701 977 flujos.
    Esta asimetria es deliberada y se declara en el articulo; conservarla es
    necesario para reproducir las cifras publicadas.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score,
                                 average_precision_score)
    import xgboost as xgb

    reg.info("=" * 78)
    reg.info("ETAPA 7/9 - BASELINES CLASICOS (RANDOM FOREST Y XGBOOST)")
    reg.info("=" * 78)

    fijar_semilla(SEMILLA)
    X = abrir_memmap(rutas)
    y_bin = np.load(rutas.finales / "y_binary.npy")
    idx_ent = np.load(rutas.finales / "idx_train.npy")
    idx_pru = np.load(rutas.finales / "idx_test.npy")

    # Extraer la submuestra estratificada de entrenamiento.
    idx_sub, _ = train_test_split(idx_ent, train_size=BASELINE_SUBMUESTRA,
                                  stratify=np.load(rutas.finales / "y_multi.npy")[idx_ent],
                                  random_state=SEMILLA)
    reg.info("  Submuestra de entrenamiento: %s flujos (%.1f %% del total)",
             f"{len(idx_sub):,}", 100 * len(idx_sub) / len(idx_ent))

    X_sub = np.asarray(X[np.sort(idx_sub)])
    y_sub = y_bin[np.sort(idx_sub)]
    X_pru = np.asarray(X[idx_pru])
    y_pru = y_bin[idx_pru]

    modelos = {
        "random_forest": RandomForestClassifier(
            n_estimators=RF_ARBOLES, max_depth=RF_PROFUNDIDAD_MAX,
            class_weight="balanced", n_jobs=-1, random_state=SEMILLA),
        "xgboost": xgb.XGBClassifier(
            n_estimators=XGB_ESTIMADORES, max_depth=XGB_PROFUNDIDAD_MAX,
            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", tree_method="hist",
            n_jobs=-1, random_state=SEMILLA),
    }

    resumen = {}
    for nombre, modelo in modelos.items():
        destino = rutas.modelos / f"resultados_{nombre}.json"
        if destino.exists() and not forzar:
            reg.info("  [omitido] los resultados de %s ya existen", nombre)
            resumen[nombre] = json.loads(destino.read_text(encoding="utf-8"))
            continue

        reg.info("  Ajustando %s ...", nombre)
        inicio = time.time()
        modelo.fit(X_sub, y_sub)
        duracion = time.time() - inicio

        puntajes = modelo.predict_proba(X_pru)[:, 1]
        pred = (puntajes >= 0.5).astype(np.int64)
        p, r, f1, _ = precision_recall_fscore_support(
            y_pru, pred, average="binary", zero_division=0)

        resultado = {
            "precision": float(p), "recall": float(r), "f1": float(f1),
            "auc_roc": float(roc_auc_score(y_pru, puntajes)),
            "auc_pr": float(average_precision_score(y_pru, puntajes)),
            "flujos_entrenamiento": int(len(idx_sub)),
            "flujos_prueba": int(len(idx_pru)),
            "segundos_entrenamiento": duracion,
            "semilla": SEMILLA,
        }
        guardar_json(resultado, destino)
        resumen[nombre] = resultado
        reg.info("  %-14s P=%.4f  R=%.4f  F1=%.4f  AUC-ROC=%.4f  AUC-PR=%.4f",
                 nombre, p, r, f1, resultado["auc_roc"], resultado["auc_pr"])

    guardar_json(resumen, rutas.tablas / "tabla8_baselines.json")


# ==============================================================================
# SECCION 12. ETAPA 8 - ABLACION DE LOS PESOS DE CLASE
# ==============================================================================

def etapa_ablacion(rutas: Rutas, reg: logging.Logger, forzar: bool) -> None:
    """
    Ejecutar las seis corridas de la ablacion: dos brazos por tres semillas.

    Los dos brazos difieren en un unico argumento, el peso pasado a la funcion
    de perdida. Arquitectura, optimizador, planificador, tamano de lote,
    criterio de parada, particiones, artefactos de preprocesamiento y orden de
    los datos son identicos, y el escalador previamente ajustado se reutiliza en
    lugar de reajustarse, de modo que la representacion de entrada es identica
    bit a bit en las seis corridas.

    Se emplean tres semillas porque el efecto debe distinguirse de la variacion
    estocastica: la Seccion 4.8 del articulo muestra que el brazo con pesos
    presenta una variabilidad entre corridas mucho mayor que el brazo sin pesos,
    y esa diferencia solo es observable con replicas.
    """
    import joblib

    reg.info("=" * 78)
    reg.info("ETAPA 8/9 - ABLACION DE LOS PESOS DE CLASE (2 brazos x 3 semillas)")
    reg.info("=" * 78)

    dispositivo = obtener_dispositivo()
    X = abrir_memmap(rutas)
    y_multi = np.load(rutas.finales / "y_multi.npy")
    idx_ent = np.load(rutas.finales / "idx_train.npy")
    idx_val = np.load(rutas.finales / "idx_val.npy")
    idx_pru = np.load(rutas.finales / "idx_test.npy")
    clases = list(joblib.load(rutas.finales / "label_encoder.pkl").classes_)
    indice_benigno = clases.index("Benign")
    indice_infiltracion = clases.index("Infilteration")

    corridas = {}
    for brazo in ABLACION_BRAZOS:
        usar_pesos = (brazo == "con_pesos")
        for semilla in ABLACION_SEMILLAS:
            etiqueta = f"{brazo}_semilla{semilla}"
            destino = rutas.ablacion / f"{etiqueta}_resultados.json"

            if destino.exists() and not forzar:
                reg.info("  [omitido] %s ya ejecutada", etiqueta)
                corridas[etiqueta] = json.loads(destino.read_text(encoding="utf-8"))
                continue

            reg.info("  Corrida %s", etiqueta)
            modelo, historial = entrenar_cnn(X, y_multi, idx_ent, idx_val, N_CLASES,
                                             semilla, usar_pesos, dispositivo, reg, etiqueta)
            resultado = evaluar_cnn(modelo, X, y_multi, idx_pru, N_CLASES,
                                    dispositivo, clases)
            resultado["historial"] = historial
            resultado["brazo"] = brazo
            resultado["semilla"] = semilla
            resultado["epocas_ejecutadas"] = len(historial)

            # Medir de forma explicita la confusion Benign -> Infilteration, que
            # es la magnitud central del hallazgo de la Seccion 4.7.
            matriz = np.asarray(resultado["matriz_confusion"])
            benignos = matriz[indice_benigno].sum()
            resultado["benigno_a_infiltracion"] = float(
                matriz[indice_benigno, indice_infiltracion] / benignos)
            resultado["recall_benigno"] = float(matriz[indice_benigno, indice_benigno] / benignos)

            import torch
            torch.save(modelo.state_dict(), rutas.ablacion / f"{etiqueta}_best.pt")
            guardar_json(resultado, destino)
            corridas[etiqueta] = resultado

    # Agregar las tres corridas de cada brazo como media y desviacion tipica.
    resumen = {}
    for brazo in ABLACION_BRAZOS:
        seleccion = [c for k, c in corridas.items() if c["brazo"] == brazo]
        def agregado(extraer):
            valores = np.array([extraer(c) for c in seleccion], dtype=np.float64)
            return {"media": float(valores.mean()), "desviacion": float(valores.std(ddof=0)),
                    "minimo": float(valores.min()), "maximo": float(valores.max())}
        resumen[brazo] = {
            "f1_macro":      agregado(lambda c: c["reporte"]["macro avg"]["f1-score"]),
            "f1_ponderado":  agregado(lambda c: c["reporte"]["weighted avg"]["f1-score"]),
            "exactitud":     agregado(lambda c: c["reporte"]["accuracy"]),
            "recall_benigno": agregado(lambda c: c["recall_benigno"]),
            "benigno_a_infiltracion": agregado(lambda c: c["benigno_a_infiltracion"]),
            "epocas": [c["epocas_ejecutadas"] for c in seleccion],
        }
        # Agregado por clase, necesario para la Tabla A3.
        resumen[brazo]["por_clase"] = {
            clase: {
                "precision": agregado(lambda c, k=clase: c["reporte"][k]["precision"]),
                "recall":    agregado(lambda c, k=clase: c["reporte"][k]["recall"]),
                "f1":        agregado(lambda c, k=clase: c["reporte"][k]["f1-score"]),
            } for clase in clases
        }

    # Cuantificar la separacion entre brazos, tal como se reporta en el articulo.
    con, sin = resumen["con_pesos"]["f1_macro"], resumen["sin_pesos"]["f1_macro"]
    desviacion_conjunta = float(np.sqrt((con["desviacion"] ** 2 + sin["desviacion"] ** 2) / 2))
    resumen["comparacion"] = {
        "diferencia_media_f1_macro": sin["media"] - con["media"],
        "brecha_entre_rangos": sin["minimo"] - con["maximo"],
        "rangos_disjuntos": bool(sin["minimo"] > con["maximo"]),
        "cohen_d": (sin["media"] - con["media"]) / desviacion_conjunta if desviacion_conjunta else None,
    }
    guardar_json(resumen, rutas.ablacion / "RESUMEN_ABLACION.json")

    reg.info("  F1-macro con pesos ... %.4f +/- %.4f", con["media"], con["desviacion"])
    reg.info("  F1-macro sin pesos ... %.4f +/- %.4f", sin["media"], sin["desviacion"])
    reg.info("  Diferencia de medias . %+.4f   Cohen d = %.1f",
             resumen["comparacion"]["diferencia_media_f1_macro"],
             resumen["comparacion"]["cohen_d"])


# ==============================================================================
# SECCION 13. ETAPA 9 - TABLAS Y FIGURAS
# ==============================================================================

def etapa_reportes(rutas: Rutas, reg: logging.Logger, forzar: bool) -> None:
    """
    Regenerar las tablas y las figuras del manuscrito a partir de los resultados
    serializados por las etapas anteriores.

    Las tablas se escriben en CSV y en Markdown; las figuras, en PNG a 300 ppp.
    Ningun valor se introduce a mano: todos proceden de los archivos JSON que
    las etapas 5 a 8 dejaron en disco.
    """
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reg.info("=" * 78)
    reg.info("ETAPA 9/9 - REGENERACION DE TABLAS Y FIGURAS")
    reg.info("=" * 78)

    leer = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
    ae = leer(rutas.modelos / "autoencoder_resultados.json")
    binaria = leer(rutas.modelos / "cnn1d_binary_resultados.json")
    multiclase = leer(rutas.modelos / "cnn1d_multiclass_resultados.json")
    ablacion = leer(rutas.ablacion / "RESUMEN_ABLACION.json")

    def escribir_tabla(nombre: str, cabecera: Sequence[str], filas: Sequence[Sequence]) -> None:
        """Volcar una tabla en CSV y en Markdown dentro del directorio de tablas."""
        with open(rutas.tablas / f"{nombre}.csv", "w", newline="", encoding="utf-8") as f:
            escritor = csv.writer(f)
            escritor.writerow(cabecera)
            escritor.writerows(filas)
        with open(rutas.tablas / f"{nombre}.md", "w", encoding="utf-8") as f:
            f.write("| " + " | ".join(map(str, cabecera)) + " |\n")
            f.write("|" + "---|" * len(cabecera) + "\n")
            for fila in filas:
                f.write("| " + " | ".join(map(str, fila)) + " |\n")
        reg.info("  [tabla] %s", nombre)

    # --- Tabla 4: metricas comparativas -------------------------------------
    p95, optimo = ae["puntos_operacion"]["p95_referencia"], ae["puntos_operacion"]["optimo"]
    bin_pos = binaria["reporte"]["Attack"]
    mc_macro = multiclase["reporte"]["macro avg"]
    escribir_tabla(
        "tabla4_metricas_comparativas",
        ["Modelo", "Umbral / tarea", "Precision", "Recall", "F1", "AUC-ROC"],
        [
            ["Autoencoder", f"p{p95['percentil']:.0f} (referencia)",
             f"{p95['precision']:.4f}", f"{p95['recall']:.4f}", f"{p95['f1']:.4f}",
             f"{ae['auc_roc']:.4f}"],
            ["Autoencoder", f"p{optimo['percentil']:.1f} (calibrado)",
             f"{optimo['precision']:.4f}", f"{optimo['recall']:.4f}", f"{optimo['f1']:.4f}",
             f"{ae['auc_roc']:.4f}"],
            ["CNN-1D", "binaria",
             f"{bin_pos['precision']:.4f}", f"{bin_pos['recall']:.4f}",
             f"{bin_pos['f1-score']:.4f}", f"{binaria['auc_roc']:.4f}"],
            ["CNN-1D", "multiclase, macro",
             f"{mc_macro['precision']:.4f}", f"{mc_macro['recall']:.4f}",
             f"{mc_macro['f1-score']:.4f}", "-"],
        ])

    # --- Tabla 5: matriz de confusion binaria -------------------------------
    mc = binaria["matriz_confusion"]
    escribir_tabla(
        "tabla5_matriz_binaria",
        ["", "Predicho: Normal", "Predicho: Ataque"],
        [["Real: Normal", f"{mc[0][0]:,}", f"{mc[0][1]:,}"],
         ["Real: Ataque", f"{mc[1][0]:,}", f"{mc[1][1]:,}"]])

    # --- Tabla A2: reporte por clase de la CNN-1D multiclase ----------------
    filas = []
    for clase, m in multiclase["reporte"].items():
        if not isinstance(m, dict) or clase in ("macro avg", "weighted avg"):
            continue
        filas.append([clase, f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                      f"{m['f1-score']:.4f}", f"{int(m['support']):,}"])
    filas.sort(key=lambda r: -int(r[4].replace(",", "")))
    filas.append(["macro avg", f"{mc_macro['precision']:.4f}", f"{mc_macro['recall']:.4f}",
                  f"{mc_macro['f1-score']:.4f}", f"{int(mc_macro['support']):,}"])
    escribir_tabla("tablaA2_reporte_multiclase",
                   ["Clase", "Precision", "Recall", "F1", "Soporte"], filas)

    # --- Tabla 7: ablacion global -------------------------------------------
    fmt = lambda d: f"{d['media']:.4f} +/- {d['desviacion']:.4f}"
    con, sin = ablacion["con_pesos"], ablacion["sin_pesos"]
    escribir_tabla(
        "tabla7_ablacion_global",
        ["Metrica", "Con pesos balanceados", "Sin pesos", "Diferencia"],
        [
            ["F1-macro", fmt(con["f1_macro"]), fmt(sin["f1_macro"]),
             f"{sin['f1_macro']['media'] - con['f1_macro']['media']:+.4f}"],
            ["F1-ponderado", fmt(con["f1_ponderado"]), fmt(sin["f1_ponderado"]),
             f"{sin['f1_ponderado']['media'] - con['f1_ponderado']['media']:+.4f}"],
            ["Exactitud", fmt(con["exactitud"]), fmt(sin["exactitud"]),
             f"{sin['exactitud']['media'] - con['exactitud']['media']:+.4f}"],
            ["Recall clase Benign", fmt(con["recall_benigno"]), fmt(sin["recall_benigno"]),
             f"{sin['recall_benigno']['media'] - con['recall_benigno']['media']:+.4f}"],
            ["Benignos etiquetados Infilteration",
             f"{100 * con['benigno_a_infiltracion']['media']:.2f} % +/- {100 * con['benigno_a_infiltracion']['desviacion']:.2f}",
             f"{100 * sin['benigno_a_infiltracion']['media']:.2f} % +/- {100 * sin['benigno_a_infiltracion']['desviacion']:.2f}",
             f"{100 * (sin['benigno_a_infiltracion']['media'] - con['benigno_a_infiltracion']['media']):+.2f} pp"],
            ["Epocas hasta la detencion temprana",
             " / ".join(map(str, con["epocas"])), " / ".join(map(str, sin["epocas"])), "-"],
        ])

    # --- Tabla A3: efecto por clase -----------------------------------------
    filas = []
    for clase in con["por_clase"]:
        c, s = con["por_clase"][clase], sin["por_clase"][clase]
        filas.append([
            clase,
            f"{c['precision']['media']:.4f} -> {s['precision']['media']:.4f}",
            f"{c['recall']['media']:.4f} -> {s['recall']['media']:.4f}",
            f"{c['f1']['media']:.4f} -> {s['f1']['media']:.4f}",
            f"{s['f1']['media'] - c['f1']['media']:+.4f}",
        ])
    escribir_tabla("tablaA3_ablacion_por_clase",
                   ["Clase", "Precision (con -> sin)", "Recall (con -> sin)",
                    "F1 (con -> sin)", "Delta F1"], filas)

    # --- Figura 1: curvas de entrenamiento ----------------------------------
    figura, ejes = plt.subplots(1, 3, figsize=(15, 4))
    ejes[0].plot([h["epoca"] for h in ae["historial"]],
                 [h["perdida_entrenamiento"] for h in ae["historial"]], label="training")
    ejes[0].plot([h["epoca"] for h in ae["historial"]],
                 [h["perdida_validacion"] for h in ae["historial"]], label="validation")
    ejes[0].set_title("Autoencoder"); ejes[0].set_xlabel("Epoch"); ejes[0].set_ylabel("MSE loss")
    ejes[0].legend()
    for eje, datos, titulo in ((ejes[1], binaria["historial"], "CNN-1D binary"),
                               (ejes[2], multiclase["historial"], "CNN-1D multiclass")):
        eje.plot([h["epoca"] for h in datos], [h["f1_macro_validacion"] for h in datos],
                 marker="o", label="validation F1-macro")
        eje.set_title(titulo); eje.set_xlabel("Epoch"); eje.set_ylabel("F1-macro")
        eje.legend()
    figura.tight_layout()
    figura.savefig(rutas.figuras / "figura1_curvas_entrenamiento.png", dpi=300)
    plt.close(figura)
    reg.info("  [figura] figura1_curvas_entrenamiento.png")

    # --- Figura 2: curva ROC y sensibilidad al umbral -----------------------
    figura, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    barrido = ae["barrido_umbral"]
    percentiles = [b["percentil"] for b in barrido]
    for clave, etiqueta in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")):
        ejes[1].plot(percentiles, [b[clave] for b in barrido], label=etiqueta)
    for percentil, estilo in ((AE_PERCENTIL_REFERENCIA, "--"), (optimo["percentil"], ":")):
        ejes[1].axvline(percentil, color="grey", linestyle=estilo)
    ejes[1].set_xlabel("Threshold percentile"); ejes[1].set_ylabel("Score"); ejes[1].legend()
    ejes[1].set_title("Threshold sensitivity")
    ejes[0].set_title(f"Autoencoder ROC (AUC = {ae['auc_roc']:.4f})")
    ejes[0].set_xlabel("False positive rate"); ejes[0].set_ylabel("True positive rate")
    figura.tight_layout()
    figura.savefig(rutas.figuras / "figura2_roc_y_umbral.png", dpi=300)
    plt.close(figura)
    reg.info("  [figura] figura2_roc_y_umbral.png")

    # --- Figura 3: matrices de confusion multiclase -------------------------
    matriz = np.asarray(multiclase["matriz_confusion"], dtype=np.float64)
    normalizada = matriz / matriz.sum(axis=1, keepdims=True)
    figura, ejes = plt.subplots(1, 2, figsize=(16, 7))
    ejes[0].imshow(np.log1p(matriz), cmap="Blues")
    ejes[0].set_title("Absolute counts (log scale)")
    imagen = ejes[1].imshow(normalizada, cmap="RdYlGn", vmin=0, vmax=1)
    ejes[1].set_title("Row-normalised (recall)")
    figura.colorbar(imagen, ax=ejes[1])
    for eje in ejes:
        eje.set_xlabel("Predicted"); eje.set_ylabel("Actual")
    figura.tight_layout()
    figura.savefig(rutas.figuras / "figura3_matrices_confusion.png", dpi=300)
    plt.close(figura)
    reg.info("  [figura] figura3_matrices_confusion.png")

    reg.info("  Tablas en %s", rutas.tablas)
    reg.info("  Figuras en %s", rutas.figuras)


# ==============================================================================
# SECCION 14. ORQUESTACION
# ==============================================================================

DESPACHADOR = {
    "descarga":    lambda r, l, f, d: etapa_descarga(r, l, f),
    "limpieza":    lambda r, l, f, d: etapa_limpieza(r, l, f),
    "escalado":    lambda r, l, f, d: etapa_escalado(r, l, f),
    "particion":   lambda r, l, f, d: etapa_particion(r, l, f),
    "autoencoder": lambda r, l, f, d: etapa_autoencoder(r, l, f, d),
    "cnn":         lambda r, l, f, d: etapa_cnn(r, l, f, d),
    "baselines":   lambda r, l, f, d: etapa_baselines(r, l, f),
    "ablacion":    lambda r, l, f, d: etapa_ablacion(r, l, f),
    "reportes":    lambda r, l, f, d: etapa_reportes(r, l, f),
}


def registrar_entorno(reg: logging.Logger) -> Dict[str, str]:
    """Registrar las versiones efectivas de las dependencias en la bitacora."""
    import platform
    import torch
    import sklearn
    import pandas as pd

    entorno = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "cuda_disponible": str(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "ninguna",
    }
    reg.info("Entorno de ejecucion:")
    for clave, valor in entorno.items():
        reg.info("  %-16s %s", clave, valor)
    return entorno


def analizar_argumentos() -> argparse.Namespace:
    """Definir e interpretar los argumentos de linea de comandos."""
    analizador = argparse.ArgumentParser(
        description="Reproduccion integra del estudio jcp-4572356 sobre CSE-CIC-IDS2018.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    analizador.add_argument("--raiz", type=Path, required=True,
                            help="directorio raiz del experimento")
    analizador.add_argument("--etapa", default="todo", choices=["todo"] + ETAPAS,
                            help="etapa a ejecutar; 'todo' encadena las nueve")
    analizador.add_argument("--forzar", action="store_true",
                            help="reejecutar la etapa descartando sus productos previos")
    analizador.add_argument("--determinista", action="store_true",
                            help="activar los nucleos deterministas de cuDNN "
                                 "(suprime la variabilidad entre corridas, "
                                 "aproximadamente 30 %% mas lento)")
    analizador.add_argument("--simular", action="store_true",
                            help="mostrar el plan de ejecucion sin ejecutar nada")
    return analizador.parse_args()


def main() -> int:
    argumentos = analizar_argumentos()
    rutas = Rutas(argumentos.raiz.expanduser().resolve())
    rutas.crear()
    reg = configurar_bitacora(rutas.bitacora)

    seleccionadas = ETAPAS if argumentos.etapa == "todo" else [argumentos.etapa]

    reg.info("=" * 78)
    reg.info("REPRODUCCION DEL ESTUDIO jcp-4572356 - CSE-CIC-IDS2018")
    reg.info("Universidad Tecnologica Israel, Quito, Ecuador")
    reg.info("=" * 78)
    reg.info("Directorio raiz .... %s", rutas.raiz)
    reg.info("Etapas ............. %s", ", ".join(seleccionadas))
    reg.info("Semilla ............ %d", SEMILLA)
    reg.info("Modo determinista .. %s", "si" if argumentos.determinista else "no")

    if argumentos.simular:
        reg.info("Modo simulacion: no se ejecuta ninguna etapa.")
        return 0

    registrar_entorno(reg)

    inicio_global = time.time()
    duraciones: Dict[str, float] = {}
    for etapa in seleccionadas:
        inicio = time.time()
        try:
            DESPACHADOR[etapa](rutas, reg, argumentos.forzar, argumentos.determinista)
        except Exception as error:                         # noqa: BLE001
            reg.error("La etapa '%s' fallo: %s", etapa, error)
            reg.error("Corregir la causa y reanudar con --etapa %s", etapa)
            raise
        duraciones[etapa] = time.time() - inicio
        reg.info("Etapa '%s' completada en %.1f min", etapa, duraciones[etapa] / 60)

    reg.info("=" * 78)
    reg.info("EJECUCION COMPLETA EN %.1f min", (time.time() - inicio_global) / 60)
    for etapa, segundos in duraciones.items():
        reg.info("  %-12s %6.1f min", etapa, segundos / 60)
    reg.info("=" * 78)
    guardar_json(duraciones, rutas.bitacora / "duraciones.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
