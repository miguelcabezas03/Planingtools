# -*- coding: utf-8 -*-
from __future__ import annotations

"""
seleccion.py — Módulo de negocio: Selección de Muestra.
"""

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from scipy.spatial import cKDTree

from logs import obtener_logger
from normalizacion import normalizar_llave
from utilidades import buscar_archivo_universo_seleccion
from validaciones import resolver_columna

ProgresoCallback = Callable[[float, str], None]

CONFIG_SELECCION_DEFECTO: dict = {
    "archivo_universo_elegible": "Universo_Elegible.xlsx",   # en carpeta Salida
    "tamano_muestra": 1636,             # T (Términos y Condiciones del país)
    "ratio_suplentes": 4,               # 1:4 (usar 5 para 1:5)
    "cuotas_gec": {"ORO": 0.25, "PLATA": 0.63, "BRONCE": 0.12},
    "tolerancia_cuotas": 0.05,
    "columna_gec": "GEC",
    "columna_ruta": "Ruta Venta",
    "columna_fijo": "Cliente fijo",
    "valor_fijo": "SI",
    "min_pdv_ruta": 5,
    "max_pdv_ruta": 12,
    "pxr_minimo": 10,
    "promedio_pdv_ruta": 10,        # define el nº de rutas: N / promedio
    "columna_estrato_geo": "Departamento",   # dispersión entre estratos
    "columna_lat": "LATITUD",
    "columna_lon": "LONGITUD",
    "limites_pais": {"lat": [-60.0, 35.0], "lon": [-120.0, -30.0]},
    "semilla": 2026,                    # null -> aleatorio en cada corrida
    "dbscan": {"eps": 0.01, "min_samples": 1},
}

PAISES_PROPORCION_4 = {
    "COSTA RICA", "ECUADOR", "EL SALVADOR", "GUATEMALA ABVO",
    "GUATEMALA EMBOCEN", "HONDURAS", "NICARAGUA", "PANAMÁ", "PANAMA",
}


# ============================================================== geografía ====
def normalizar_coordenadas(
    col_a: pd.Series, col_b: pd.Series, limites: dict
) -> tuple[pd.Series, pd.Series]:
    """
    Repara coordenadas con columnas intercambiadas y escalas 10^k mezcladas
    (p. ej. LATITUD=-894.38 que en realidad es longitud -89.438).
    Prueba, por fila, ambas asignaciones y escalas 10^0..10^6 hasta caer en
    los límites del país. Devuelve (lat, lon) con NaN donde no fue posible.
    """
    def a_num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(
            s.astype(str).str.replace(",", ".", regex=False).str.strip(),
            errors="coerce",
        )

    def escalar(v: pd.Series, lo: float, hi: float) -> pd.Series:
        r = pd.Series(np.nan, index=v.index)
        for k in range(7):
            c = v / (10 ** k)
            m = r.isna() & c.between(lo, hi)
            r[m] = c[m]
        return r

    a, b = a_num(col_a), a_num(col_b)
    lat_lo, lat_hi = limites["lat"]
    lon_lo, lon_hi = limites["lon"]

    la1, lo1 = escalar(a, lat_lo, lat_hi), escalar(b, lon_lo, lon_hi)   # A=lat
    la2, lo2 = escalar(b, lat_lo, lat_hi), escalar(a, lon_lo, lon_hi)   # A=lon
    usar_swap = la2.notna() & lo2.notna()
    lat = la2.where(usar_swap, la1)
    lon = lo2.where(usar_swap, lo1)
    return lat, lon


def haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Distancia haversine en km (vectorizada, difunde formas de numpy)."""
    r = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dla, dlo = la2 - la1, lo2 - lo1
    h = np.sin(dla / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlo / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(h))


def dbscan_coordenadas(
    latitud: np.ndarray,
    longitud: np.ndarray,
    eps: float,
    min_samples: int,
) -> np.ndarray:
    """DBSCAN euclidiano sin almacenar todos los vecinos en memoria."""
    coordenadas = np.column_stack((latitud, longitud)).astype(float)
    if len(coordenadas) == 0:
        return np.empty(0, dtype=int)
    if not np.isfinite(coordenadas).all():
        raise ErrorSeleccion("La clusterización recibió coordenadas no válidas.")
    if eps <= 0 or min_samples <= 0:
        raise ErrorSeleccion("DBSCAN requiere EPS y Min Samples mayores que cero.")

    radio_cuadrado = float(eps) * float(eps)
    tolerancia = np.finfo(float).eps * max(1.0, radio_cuadrado) * 8
    radio = math.sqrt(radio_cuadrado + tolerancia)
    arbol = cKDTree(coordenadas)
    es_nucleo = arbol.query_ball_point(coordenadas, radio, return_length=True) >= min_samples
    etiquetas = np.full(len(coordenadas), -1, dtype=int)
    nucleos_procesados = np.zeros(len(coordenadas), dtype=bool)
    cluster = 0
    for inicio in range(len(coordenadas)):
        if not es_nucleo[inicio] or nucleos_procesados[inicio]:
            continue
        pendientes = [inicio]
        nucleos_procesados[inicio] = True
        etiquetas[inicio] = cluster
        while pendientes:
            actual = pendientes.pop()
            for vecino in sorted(arbol.query_ball_point(coordenadas[actual], radio)):
                if etiquetas[vecino] == -1:
                    etiquetas[vecino] = cluster
                if es_nucleo[vecino] and not nucleos_procesados[vecino]:
                    nucleos_procesados[vecino] = True
                    pendientes.append(vecino)
        cluster += 1
    return etiquetas


# ================================================================ resultado ==
@dataclass
class ResultadoSeleccion:
    titulares: pd.DataFrame = field(default_factory=pd.DataFrame)
    suplentes: pd.DataFrame = field(default_factory=pd.DataFrame)
    universo_revisado: pd.DataFrame = field(default_factory=pd.DataFrame)
    resumen_rutas: pd.DataFrame = field(default_factory=pd.DataFrame)
    resumen_gec: pd.DataFrame = field(default_factory=pd.DataFrame)
    resumen_canal: pd.DataFrame = field(default_factory=pd.DataFrame)
    metricas: dict = field(default_factory=dict)
    inicio: datetime = field(default_factory=datetime.now)
    duracion_seg: float = 0.0


class ErrorSeleccion(Exception):
    """Error de negocio de la selección, apto para ventana emergente."""


# ==================================================================== motor ==
class SelectorMuestra:
    """Motor de selección estratificada con agrupación por ruta y suplentes."""

    def __init__(self, carpeta_salida: Path, cfg: dict) -> None:
        self.salida = carpeta_salida
        self.cfg = {**CONFIG_SELECCION_DEFECTO, **(cfg or {})}
        self.log = obtener_logger()
        self.rng = np.random.default_rng(self.cfg.get("semilla"))

    # ------------------------------------------------------------------ API --
    def cargar_universo(self) -> pd.DataFrame:
        """Lee el universo correspondiente al país sin ejecutar la selección."""
        pais_act = getattr(self, "pais_activo", self.cfg.get("pais_activo", ""))
        rutas_map = getattr(self, "rutas_app", {})
        dir_sel = rutas_map.get("entrada_seleccion", self.salida)
        dir_dep = rutas_map.get("salida_depuracion", self.salida)
        ruta_final, tokens = buscar_archivo_universo_seleccion(dir_sel, dir_dep, pais_act)
        if not ruta_final:
            raise ErrorSeleccion(
                f"No se encontró el archivo del universo elegible para '{pais_act}' "
                "en la carpeta 'Entrada Seleccion'.\n\n"
                "Asegúrese de colocar el archivo Excel en 'Entrada Seleccion' con "
                f"algún identificador como ({', '.join(tokens)})."
            )
        try:
            uni = pd.read_excel(ruta_final, engine="openpyxl")
        except PermissionError as exc:
            raise ErrorSeleccion(
                f"El archivo '{ruta_final.name}' está abierto por otro usuario o "
                "programa.\n\nPor favor ciérrelo en Excel e inténtelo nuevamente."
            ) from exc
        uni.columns = [c.strip() if isinstance(c, str) else c for c in uni.columns]
        self.log.info(
            "Selección: universo leído desde '%s' con %s tiendas.",
            ruta_final.name, f"{len(uni):,}",
        )
        self.ruta_universo = ruta_final
        return uni

    def ejecutar(
        self,
        progreso: ProgresoCallback | None = None,
        universo: pd.DataFrame | None = None,
        fijos: pd.DataFrame | None = None,
    ) -> ResultadoSeleccion:
        avisar = progreso or (lambda p, m: None)
        t0 = time.perf_counter()
        cfg, log = self.cfg, self.log
        res = ResultadoSeleccion()

        # 1) Insumo: universo elegible (resolución dinámica por país) -------
        avisar(0.05, "Leyendo universo elegible...")
        pais_act = getattr(self, "pais_activo", cfg.get("pais_activo", ""))
        res.pais_activo = pais_act
        cfg["pais_activo"] = pais_act
        res.nueva_regla = pais_act.strip().upper() in PAISES_PROPORCION_4
        if res.nueva_regla:
            cfg["ratio_suplentes"] = 0
        cfg["min_pdv_ruta"] = max(10, int(cfg.get("min_pdv_ruta", 10)))
        cfg["max_pdv_ruta"] = max(10, int(cfg.get("max_pdv_ruta", 10)))
        res.pais_activo = pais_act
        uni = universo.copy() if universo is not None else self.cargar_universo()
        uni.columns = [c.strip() if isinstance(c, str) else c for c in uni.columns]
        self._resolver_columnas_configuradas(uni)
        uni = self._incorporar_fijos(uni, fijos)

        faltantes = [c for c in (cfg.get("columna_gec"), cfg.get("columna_lat"), cfg.get("columna_lon")) if c and c not in uni.columns]
        if cfg.get("columna_ruta") and cfg["columna_ruta"] not in uni.columns:
            faltantes.append(cfg["columna_ruta"])
        if faltantes:
            raise ErrorSeleccion(
                f"Al universo elegible le faltan columnas: {faltantes}."
            )
        uni = self._aplicar_regla_pxr(uni)
        if "Cupo_Restante" not in uni.columns:
            log.warning("Selección: no hay 'Cupo_Restante'; prioridad uniforme.")
            uni["Cupo_Restante"] = 1
        res.universo_revisado = uni.copy()
        res.universo_revisado["seleccion"] = ""

        estado = normalizar_llave(uni["ELEGIBLE"])
        col_fijo = cfg.get("columna_fijo")
        es_fijo = (
            uni["_FIJO_ARCHIVO"].astype(bool)
            if fijos is not None else
            self._normalizar_tipo(uni[col_fijo]) == "FIJO"
            if col_fijo in uni.columns else pd.Series(False, index=uni.index)
        )
        # El archivo de Fijos es la lista obligatoria que sube el usuario.
        uni_elegible = uni.loc[(estado == "ELEGIBLE") | es_fijo].copy()
        if uni_elegible.empty:
            raise ErrorSeleccion(
                "No quedaron puntos ELEGIBLES después de la revisión geográfica y la regla PXR."
            )

        # 2) Coordenadas ------------------------------------------------------
        avisar(0.15, "Normalizando coordenadas GPS...")
        limites = cfg.get("limites_pais", {"lat": [-60.0, 35.0], "lon": [-120.0, -30.0]})
        uni_elegible["_LAT"], uni_elegible["_LON"] = normalizar_coordenadas(
            uni_elegible[cfg["columna_lat"]], uni_elegible[cfg["columna_lon"]], limites
        )
        sin_gps = int(uni_elegible["_LAT"].isna().sum())
        log.info("Selección: %s tiendas sin GPS válido (solo los fijos son seleccionables).",
                 f"{sin_gps:,}")
        con_gps = uni_elegible["_LAT"].notna() & uni_elegible["_LON"].notna()
        pool = uni_elegible.loc[con_gps].copy()
        pool["_GEC"] = normalizar_llave(pool[cfg["columna_gec"]])
        
        if cfg.get("columna_ruta") and cfg["columna_ruta"] in pool.columns:
            pool["_RUTA"] = normalizar_llave(pool[cfg["columna_ruta"]])
        elif cfg.get("columna_agencia") and cfg["columna_agencia"] in pool.columns:
            pool["_RUTA"] = normalizar_llave(pool[cfg["columna_agencia"]])
        else:
            pool["_RUTA"] = "R1"

        # Los notebooks de CH y RD eliminan rutas demasiado pequeñas antes de
        # clusterizar. PXR normalmente ya hizo este filtro, pero se conserva
        # aquí para que la selección también funcione con un universo externo.
        pais_norm = pais_act.strip().upper()
        if pais_norm in {"CHILE", "REPÚBLICA DOMINICANA", "REPUBLICA DOMINICANA"}:
            minimo_base = int(cfg.get("puntos_rutas", {}).get("min_ruta_base", 0))
            if minimo_base > 1:
                tamanos_ruta = pool.groupby("_RUTA")["_RUTA"].transform("size")
                descartados = int((tamanos_ruta < minimo_base).sum())
                pool = pool.loc[(tamanos_ruta >= minimo_base) | es_fijo.reindex(pool.index, fill_value=False)].copy()
                log.info(
                    "Selección: %s puntos retirados de rutas con menos de %s PDV.",
                    f"{descartados:,}", minimo_base,
                )
        avisar(0.24, "Clusterizando puntos y calculando densidad...")
        pool = self._preparar_clusterizacion(pool) if not pool.empty else pool
        fijos_sin_gps = uni_elegible.loc[~con_gps & es_fijo.reindex(uni_elegible.index, fill_value=False)].copy()
        if not fijos_sin_gps.empty:
            fijos_sin_gps["_GEC"] = normalizar_llave(fijos_sin_gps[cfg["columna_gec"]])
            columna_ruta = cfg.get("columna_ruta") or cfg.get("columna_agencia")
            fijos_sin_gps["_RUTA"] = normalizar_llave(fijos_sin_gps[columna_ruta]) if columna_ruta in fijos_sin_gps else "R1"
            fijos_sin_gps["_CANAL"] = (
                self._normalizar_canal(fijos_sin_gps[cfg["columna_canal"]])
                if cfg.get("columna_canal") in fijos_sin_gps.columns else ""
            )
            fijos_sin_gps["_TIPO"] = "FIJO"
            fijos_sin_gps["CLUSTER"] = -1
            fijos_sin_gps["DENSIDAD"] = 0
            fijos_sin_gps["_PUNTAJE_CLUSTER"] = 0.0
            fijos_sin_gps["PESO_CELDA_NORM"] = 0.0
            pool = pd.concat([pool, fijos_sin_gps], axis=0)
        pool["_FIJO_FORZADO"] = es_fijo.reindex(pool.index, fill_value=False).astype(bool)
        if fijos is not None:
            pool["_TIPO"] = np.where(pool["_FIJO_FORZADO"], "FIJO", "VARIABLE")
        if pool.empty:
            raise ErrorSeleccion("No hay puntos seleccionables ni fijos cargados.")

        # 3) Titulares --------------------------------------------------------
        avisar(0.30, "Seleccionando titulares (T)...")
        titulares = self._seleccionar_titulares(pool)
        res.universo_revisado.loc[titulares.index, "seleccion"] = "T"
        res.resumen_rutas, res.resumen_gec, res.resumen_canal = self._resumenes_seleccion(pool, titulares)

        # 4) Dispersión -------------------------------------------------------
        avisar(0.60, "Calculando dispersión (distancias entre T)...")
        titulares = self._calcular_dispersion(titulares)

        # 5) Suplentes --------------------------------------------------------
        avisar(0.75, "Asignando suplentes S1..Sn...")
        suplentes = self._asignar_suplentes(titulares, pool)

        # 6) Salida -----------------------------------------------------------
        avisar(0.92, "Preparando archivos de muestra...")
        res.titulares = self._presentar(titulares, es_titular=True)
        res.suplentes = self._presentar(suplentes, es_titular=False)
        res.duracion_seg = time.perf_counter() - t0
        res.metricas = self._metricas(uni_elegible, pool, res, sin_gps)
        res.metricas["Fijos sin GPS incluidos"] = len(fijos_sin_gps)
        return res

    def _incorporar_fijos(self, universo: pd.DataFrame, fijos: pd.DataFrame | None) -> pd.DataFrame:
        """Marca por código todos los fijos cargados y agrega los ausentes."""
        out = universo.reset_index(drop=True).copy()
        out["_FIJO_ARCHIVO"] = False
        if fijos is None or fijos.empty:
            return out
        fijos = fijos.copy()
        fijos.columns = [c.strip() if isinstance(c, str) else c for c in fijos.columns]
        codigo = self.cfg.get("llave_universo") or self.cfg.get("columna_codigo")
        candidatos = [codigo, "COD D&N", "Codigo D&N", "CÓDIGO", "CODIGO", "Codigo", "RefID", "RefIDEmbotellador"]
        col_uni = resolver_columna(out, [nombre for nombre in candidatos if nombre])
        col_fijos = resolver_columna(fijos, [nombre for nombre in [col_uni] + candidatos if nombre])
        if not col_uni or not col_fijos:
            raise ErrorSeleccion("El archivo Fijos requiere la columna de código configurada en el universo.")
        cod_uni = normalizar_llave(out[col_uni])
        cod_fijos = normalizar_llave(fijos[col_fijos])
        if cod_fijos.eq("").any():
            raise ErrorSeleccion("Hay puntos en el archivo Fijos sin código; no se pueden identificar todos.")
        fijos = fijos.loc[~cod_fijos.duplicated()].copy()
        cod_fijos = cod_fijos.loc[fijos.index]
        out.loc[cod_uni.isin(set(cod_fijos)), "_FIJO_ARCHIVO"] = True
        faltantes = fijos.loc[~cod_fijos.isin(set(cod_uni))].copy()
        if not faltantes.empty:
            if col_fijos != col_uni:
                faltantes[col_uni] = faltantes[col_fijos]
            faltantes["_FIJO_ARCHIVO"] = True
            faltantes.index = range(len(out), len(out) + len(faltantes))
            out = pd.concat([out, faltantes], axis=0)
        col_tipo = self.cfg.get("columna_fijo")
        if col_tipo:
            out.loc[out["_FIJO_ARCHIVO"], col_tipo] = self.cfg.get("valor_fijo", "SI")
        return out

    def _aplicar_regla_pxr(self, universo: pd.DataFrame) -> pd.DataFrame:
        """Agrega PXR (conteo de código por ruta) y excluye grupos pequeños."""
        out = universo.copy()
        cfg = self.cfg
        self._resolver_columnas_configuradas(out)
        columna_ruta = cfg.get("columna_ruta") or cfg.get("columna_agencia")
        if not columna_ruta or columna_ruta not in out.columns:
            raise ErrorSeleccion(
                f"No se encontró la columna de ruta configurada: '{columna_ruta}'."
            )
        if "ELEGIBLE" not in out.columns:
            out["ELEGIBLE"] = "ELEGIBLE"

        candidatos_codigo = [
            cfg.get("columna_codigo"), "Codigo D&N", "Codigo DN",
            "CÓDIGO", "CODIGO", "Código", "RefID", "RefIDEmbotellador",
            "RefIDBase (d&n)", "ID cliente/PDV", "NSR Client ID",
        ]
        columna_codigo = next(
            (c for c in candidatos_codigo if c and c in out.columns), None,
        )
        try:
            minimo = max(1, int(cfg.get("pxr_minimo", 10)))
        except (TypeError, ValueError):
            minimo = 10
        elegible = normalizar_llave(out["ELEGIBLE"]) == "ELEGIBLE"
        obligatorios = out.get("_FIJO_ARCHIVO", pd.Series(False, index=out.index)).fillna(False).astype(bool)
        contables = elegible | obligatorios
        agrupador = out[columna_ruta].fillna("(SIN RUTA)").astype(str)
        out["PXR"] = pd.Series(0, index=out.index, dtype="Int64")
        if contables.any():
            universo_elegible = out.loc[contables]
            grupos_elegibles = agrupador.loc[contables]
            if columna_codigo:
                conteos = universo_elegible.groupby(
                    grupos_elegibles, dropna=False
                )[columna_codigo].transform("count")
            else:
                self.log.warning(
                    "PXR: no se encontró la columna de código; se usará el tamaño de la ruta."
                )
                conteos = universo_elegible.groupby(
                    grupos_elegibles, dropna=False
                )[columna_ruta].transform("size")
            out.loc[contables, "PXR"] = conteos.astype("Int64")

        excluir = elegible & (out["PXR"] < minimo)
        out.loc[excluir, "ELEGIBLE"] = f"NO ELEGIBLE PXR <{minimo}"
        self.log.info(
            "PXR: %s tiendas excluidas por rutas con menos de %s puntos.",
            f"{int(excluir.sum()):,}", minimo,
        )
        return out

    def _resolver_columnas_configuradas(self, datos: pd.DataFrame) -> None:
        """Adapta mayúsculas y alias habituales sin alterar el archivo original."""
        alias = {
            "columna_gec": ["GEC", "Gec", "TAMAÑO ICE KO", "CLASIFICACION ICE"],
            "columna_ruta": ["Ruta", "RUTA", "Ruta Venta", "RUTA PREVENTA", "RutaEmbotellador"],
            "columna_lat": ["Latitud", "LATITUD", "LATITUDE", "Lat", "LAT"],
            "columna_lon": ["Longitud", "LONGITUD", "LONGITUDE", "Lon", "LON"],
            "columna_fijo": ["PDV FIJO/PRIORITARIO", "Cliente fijo", "CLIENTE FIJO 30%", "FIJO"],
            "columna_canal": ["Canal", "Canal País", "TIPO CLIENTE ICE (D&N)"],
            "columna_region": ["Region", "REGION", "Región"],
            "columna_subcanal": ["SubCanal", "SUBCANAL", "SUB CANAL"],
            "columna_agencia": ["Agencia", "AGENCIA", "LOCALIDAD-AGENCIA"],
            "columna_peso": ["PESO_CELDA", "Peso", "PESO"],
        }
        for clave, opciones in alias.items():
            configurada = self.cfg.get(clave)
            if not configurada:
                continue
            candidatos = [configurada] + [v for v in opciones if v != configurada]
            encontrada = resolver_columna(datos, candidatos)
            if encontrada:
                self.cfg[clave] = encontrada

    @staticmethod
    def _normalizar_canal(serie: pd.Series) -> pd.Series:
        valores = normalizar_llave(serie)
        salida = valores.copy()
        es_on = valores.isin(["ON", "ON PREMISE", "PREMISE", "RESTAURANTE"])
        es_off = valores.isin(["OFF", "HOME MARKET TRADICIONAL", "BODEGA"])
        salida.loc[es_on] = "ON"
        salida.loc[es_off] = "OFF"
        return salida

    @staticmethod
    def _normalizar_tipo(serie: pd.Series) -> pd.Series:
        valores = normalizar_llave(serie)
        salida = valores.copy()
        salida.loc[valores.isin(["FIJO", "SI", "SÍ", "YES", "TRUE", "1"])] = "FIJO"
        salida.loc[valores.isin(["VARIABLE", "NO", "FALSE", "0"])] = "VARIABLE"
        return salida

    def _preparar_clusterizacion(self, pool: pd.DataFrame) -> pd.DataFrame:
        """Añade CLUSTER, DENSIDAD y el puntaje geográfico del optimizador."""
        cfg = self.cfg
        salida = pool.copy()
        parametros = cfg.get("dbscan") or cfg.get("cluster") or {}
        eps = float(parametros.get("eps", 0.01))
        min_samples = int(parametros.get("min_samples", 1))
        etiquetas = dbscan_coordenadas(
            salida["_LAT"].to_numpy(float),
            salida["_LON"].to_numpy(float),
            eps,
            min_samples,
        )

        # Ecuador conserva el ruido creando una ruta natural individual. Para
        # los demás países esta regla evita perder elegibles si min_samples>1.
        ruido = etiquetas == -1
        if ruido.any():
            siguiente = int(etiquetas[~ruido].max() + 1) if (~ruido).any() else 0
            etiquetas[ruido] = np.arange(siguiente, siguiente + int(ruido.sum()))
        salida["CLUSTER"] = etiquetas
        salida["DENSIDAD"] = salida.groupby("CLUSTER")["CLUSTER"].transform("size").astype(int)

        col_canal = cfg.get("columna_canal")
        col_fijo = cfg.get("columna_fijo")
        salida["_CANAL"] = (
            self._normalizar_canal(salida[col_canal])
            if col_canal and col_canal in salida.columns
            else ""
        )
        salida["_TIPO"] = (
            self._normalizar_tipo(salida[col_fijo])
            if col_fijo and col_fijo in salida.columns
            else "VARIABLE"
        )

        columnas_celda = ["_RUTA", "_CANAL", "_GEC"]
        salida["PESO_CELDA"] = salida.groupby(columnas_celda)["_RUTA"].transform("size").astype(int)
        salida["PESO_CELDA_LOG"] = np.log1p(salida["PESO_CELDA"].astype(float))
        maximo_log = float(salida["PESO_CELDA_LOG"].max())
        salida["PESO_CELDA_NORM"] = (
            salida["PESO_CELDA_LOG"] / maximo_log if maximo_log > 0 else 0.0
        )

        pais = str(cfg.get("pais_activo", "")).strip().upper()
        if pais == "ECUADOR":
            salida["_PUNTAJE_CLUSTER"] = salida["DENSIDAD"].astype(float) ** 2
        elif pais in {"REPÚBLICA DOMINICANA", "REPUBLICA DOMINICANA"}:
            salida["_PUNTAJE_CLUSTER"] = (
                salida["DENSIDAD"].astype(float) * 0.3
                + salida["PESO_CELDA_NORM"] * 0.5
                + (salida["_CANAL"] == "ON").astype(float) * 0.2
            )
        else:
            salida["_PUNTAJE_CLUSTER"] = salida["DENSIDAD"].astype(float)
        self.log.info(
            "Clusterización DBSCAN: eps=%s, min_samples=%s, clusters=%s, densidad máxima=%s.",
            eps, min_samples, salida["CLUSTER"].nunique(), salida["DENSIDAD"].max(),
        )
        return salida

    # ------------------------------------------------------------ titulares --

    def _seleccionar_titulares_ortools(self, pool: pd.DataFrame) -> pd.DataFrame | None:
        """
        Ejecuta el modelo exacto de Programación Lineal Entera (ILP) con OR-Tools SCIP
        utilizado en los cuadernos de ASIGNACIONES para Chile, Ecuador, RD, GT AB, GT EM, etc.
        """
        cfg, log = self.cfg, self.log
        
        pais_act = cfg.get("pais_activo", "").strip().upper()
        # La proporción x4 usa el motor heurístico; este solver conserva el
        # modelo histórico para los llamados directos y Chile/RD.
        nueva_regla = pais_act not in {"REPÚBLICA DOMINICANA", "REPUBLICA DOMINICANA", "ECUADOR", "CHILE"}
        
        tamano_base = int(cfg.get("tamano_muestra", len(pool)))
        fijos_obligatorios = pool.get("_FIJO_FORZADO", pd.Series(False, index=pool.index)).fillna(False)
        n_total = max(tamano_base * 4 if nueva_regla else tamano_base, int(fijos_obligatorios.sum()))
        
        # 1. Preparar solver SCIP
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            log.warning("OR-Tools SCIP no disponible, recurriendo a motor heurístico.")
            return None
            
        # Los modelos de referencia de Chile trabajan hasta 60 segundos. Los
        # tres países auditados usan ese margen para optimizar la densidad.
        solver.set_time_limit(60000 if not nueva_regla else 15000)
        
        # Variables de decisión binarias por PDV
        x = {i: solver.IntVar(0, 1, f"x_{i}") for i in pool.index}
        for i in pool.index[fijos_obligatorios]:
            solver.Add(x[i] == 1)
        
        # Restricción 1: Tamaño Total Muestra
        if nueva_regla:
            solver.Add(solver.Sum(x[i] for i in pool.index) <= n_total)
        else:
            solver.Add(solver.Sum(x[i] for i in pool.index) == n_total)
        
        # Restricción 2: Cuotas GEC
        cuotas_gec = cfg.get("cuotas_gec", {})
        for gec_cat, val in cuotas_gec.items():
            gec_cat_norm = str(gec_cat).strip().upper()
            idx_gec = [i for i in pool.index if pool.loc[i, "_GEC"] == gec_cat_norm]
            target_v = int(val) if val >= 1 else int(round(n_total * val))
            if nueva_regla:
                target_v_4 = int(target_v * 4 if val >= 1 else round(tamano_base * 4 * val))
                target_v_6 = int(target_v * 4.35 if val >= 1 else round(tamano_base * 4.35 * val))
                solver.Add(solver.Sum(x[i] for i in idx_gec) <= min(target_v_6, len(idx_gec)))
            else:
                solver.Add(solver.Sum(x[i] for i in idx_gec) == target_v)
            
        # Restricción 3: Cuotas Canal (ON / OFF)
        cuotas_canal = cfg.get("cuotas_canal", {})
        col_canal = cfg.get("columna_canal")
        if col_canal and col_canal in pool.columns:
            for cat, val in cuotas_canal.items():
                cat_n = self._normalizar_canal(pd.Series([cat])).iloc[0]
                idx_canal = pool.index[pool["_CANAL"] == cat_n].tolist()
                target_v = int(val) if val >= 1 else int(round(n_total * val))
                if nueva_regla:
                    target_v_6 = int(target_v * 4.35 if val >= 1 else round(tamano_base * 4.35 * val))
                    solver.Add(solver.Sum(x[i] for i in idx_canal) <= min(target_v_6, len(idx_canal)))
                else:
                    solver.Add(solver.Sum(x[i] for i in idx_canal) == target_v)

        # Restricción 4: Cuotas Tipo/Fijo (FIJO / VARIABLE)
        cuotas_tipo = cfg.get("cuotas_tipo", {})
        col_fijo = cfg.get("columna_fijo")
        if col_fijo and col_fijo in pool.columns and not fijos_obligatorios.any():
            for cat, val in cuotas_tipo.items():
                cat_n = self._normalizar_tipo(pd.Series([cat])).iloc[0]
                idx_t = pool.index[pool["_TIPO"] == cat_n].tolist()
                target_v = int(val) if val >= 1 else int(round(n_total * val))
                if nueva_regla:
                    if cat_n == "FIJO":
                        target_v_4 = min(target_v, len(idx_t))
                        solver.Add(solver.Sum(x[i] for i in idx_t) == target_v_4)
                    else:
                        target_v_10 = int(target_v * 10.0 if val >= 1 else round(tamano_base * 10.0 * val))
                        solver.Add(solver.Sum(x[i] for i in idx_t) <= min(target_v_10, len(idx_t)))
                else:
                    solver.Add(solver.Sum(x[i] for i in idx_t) == target_v)

        # Restricción 5: Chile Regiones
        cuotas_region = cfg.get("cuotas_region", {})
        col_region = cfg.get("columna_region") or ("Region" if "Region" in pool.columns else ("REGIONAL" if "REGIONAL" in pool.columns else None))
        if cuotas_region and col_region and col_region in pool.columns:
            reg_norm = normalizar_llave(pool[col_region])
            for reg, val in cuotas_region.items():
                idx_reg = pool[reg_norm == str(reg).strip().upper()].index
                solver.Add(solver.Sum(x[i] for i in idx_reg) == int(val))

        # Restricción 6: Chile Subcanales
        cuotas_subcanal = cfg.get("cuotas_subcanal", {})
        col_subcanal = cfg.get("columna_subcanal") or ("Subcanal" if "Subcanal" in pool.columns else ("SUBCANAL" if "SUBCANAL" in pool.columns else None))
        if cuotas_subcanal and col_subcanal and col_subcanal in pool.columns:
            sub_norm = normalizar_llave(pool[col_subcanal])
            for sub, val in cuotas_subcanal.items():
                idx_sub = pool[sub_norm == str(sub).strip().upper()].index
                solver.Add(solver.Sum(x[i] for i in idx_sub) == int(val))

        # Restricción 7: Ecuador Agencias
        cuotas_agencia = cfg.get("cuotas_agencia", {})
        col_agencia = cfg.get("columna_agencia") or ("Agencia" if "Agencia" in pool.columns else ("AGENCIA" if "AGENCIA" in pool.columns else None))
        if cuotas_agencia and col_agencia and col_agencia in pool.columns:
            ag_norm = normalizar_llave(pool[col_agencia])
            for ag, val in cuotas_agencia.items():
                idx_ag = pool[ag_norm == str(ag).strip().upper()].index
                solver.Add(solver.Sum(x[i] for i in idx_ag) == int(val))

        # Restricción 8: RD Fijo x Canal
        cuotas_fijo_canal = cfg.get("cuotas_fijo_canal", {})
        if cuotas_fijo_canal and col_fijo and col_canal and col_fijo in pool.columns and col_canal in pool.columns:
            for clave, val in cuotas_fijo_canal.items():
                if isinstance(clave, (tuple, list)) and len(clave) == 2:
                    f_type, c_type = clave
                else:
                    partes = str(clave).split("_", 1)
                    if len(partes) != 2:
                        raise ErrorSeleccion(
                            f"Cuota fijo/canal inválida: '{clave}'. Use, por ejemplo, FIJO_OFF."
                        )
                    f_type, c_type = partes
                tipo_obj = self._normalizar_tipo(pd.Series([f_type])).iloc[0]
                canal_obj = self._normalizar_canal(pd.Series([c_type])).iloc[0]
                mask = (pool["_TIPO"] == tipo_obj) & (pool["_CANAL"] == canal_obj)
                idx_fc = pool[mask].index
                solver.Add(solver.Sum(x[i] for i in idx_fc) == int(val))

        # Restricción 9: Códigos Prioritarios (Guatemala ABVO)
        codigos_prio = cfg.get("codigos_prioritarios", [])
        if codigos_prio:
            col_cod = next((c for c in [cfg.get("llave_universo"), "CÓDIGO", "CODIGO", "Codigo", "RefIDEmbotellador"] if c and c in pool.columns), None)
            if col_cod:
                s_cod = normalizar_llave(pool[col_cod])
                set_prio = {str(z).strip() for z in codigos_prio}
                idx_prio = pool[s_cod.isin(set_prio)].index
                for i in idx_prio:
                    solver.Add(x[i] == 1)

        # Restricciones por Ruta / Agrupación
        col_ruta = cfg.get("columna_ruta") if cfg.get("columna_ruta") in pool.columns else (col_agencia if col_agencia and col_agencia in pool.columns else None)
        if col_ruta:
            minr = int(cfg.get("min_pdv_ruta", 1))
            maxr = int(cfg.get("max_pdv_ruta", 200))
            rutas_grouped = pool.groupby("_RUTA").groups
            for r_name, r_indices in rutas_grouped.items():
                n_fijos = int(fijos_obligatorios.loc[r_indices].sum())
                y_r = solver.IntVar(0, 1, f"y_{len(str(r_name))}_{str(r_name)}")
                if n_fijos:
                    solver.Add(y_r == 1)
                minimo_factible = min(minr, len(r_indices)) if n_fijos else minr
                solver.Add(solver.Sum(x[i] for i in r_indices) >= minimo_factible * y_r)
                solver.Add(solver.Sum(x[i] for i in r_indices) <= max(maxr, n_fijos) * y_r)

        # Función Objetivo Multi-etapa (Maximizar Fijos + Peso Celda / Densidad)
        peso_norm = pool.get("PESO_CELDA_NORM", pd.Series(1.0, index=pool.index))
        fijo_factor = (pool["_TIPO"] == "FIJO").astype(float) + 1.0
        
        if nueva_regla:
            # Priorizar ORO > PLATA > BRONCE, ON > OFF
            gec_scores = {"ORO": 10000.0, "PLATA": 1000.0, "BRONCE": 10.0}
            canal_col = cfg.get("columna_canal")
            has_canal = canal_col and canal_col in pool.columns
            canal_series = normalizar_llave(pool[canal_col]) if has_canal else None
            
            def get_weight(i):
                w = 1000.0
                g = str(pool.loc[i, "_GEC"]).strip().upper()
                w += gec_scores.get(g, 0.0)
                if has_canal:
                    c = str(canal_series.loc[i]).strip().upper()
                    if c in ["ON", "ON PREMISE"]: w += 5000.0
                    elif c in ["OFF", "HOME MARKET TRADICIONAL", "BODEGA"]: w += 100.0
                w += float(fijo_factor.loc[i]) * 50000.0 # Fijos ultra prioritarios
                w += float(peso_norm.loc[i]) * 10.0
                return w
                
            solver.Maximize(solver.Sum(x[i] * get_weight(i) for i in pool.index))
        else:
            # En CH, RD y EC las cuotas son restricciones duras. Entre todas
            # las soluciones válidas se priorizan los núcleos más agrupados,
            # siguiendo el objetivo geográfico de sus notebooks.
            puntaje = pool.get("_PUNTAJE_CLUSTER", pool["DENSIDAD"].astype(float))
            solver.Maximize(solver.Sum(x[i] * float(puntaje.loc[i]) for i in pool.index))
        
        status = solver.Solve()
        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            sel_idx = [i for i in pool.index if x[i].solution_value() > 0.5]
            log.info("OR-Tools SCIP Solver encontró solución factible con %s titulares!", len(sel_idx))
            return pool.loc[sel_idx].copy()
        else:
            log.warning("OR-Tools SCIP Solver no halló solución factible (Status %s), usando motor heurístico.", status)
            return None

    def _seleccionar_titulares(self, pool: pd.DataFrame) -> pd.DataFrame:
        cfg, log = self.cfg, self.log
        pais_act = cfg.get("pais_activo", "").strip().upper()
        nueva_regla = pais_act in PAISES_PROPORCION_4
        if nueva_regla:
            return self._seleccionar_titulares_proporcion(pool)
        tamano_base = int(cfg["tamano_muestra"])
        fijos_obligatorios = pool["_FIJO_FORZADO"].fillna(False) | pool["_TIPO"].eq("FIJO")
        n_total = max(tamano_base, int(fijos_obligatorios.sum()))
        
        # Intentar solver exacto ILP OR-Tools SCIP de ASIGNACIONES primero
        res_ortools = self._seleccionar_titulares_ortools(pool)
        if res_ortools is not None:
            if nueva_regla and len(res_ortools) > 0:
                log.info(f"OR-Tools devolvió {len(res_ortools)} titulares (objetivo: {n_total})")
                return res_ortools
            elif len(res_ortools) == n_total:
                return res_ortools
        minr, maxr = int(cfg["min_pdv_ruta"]), int(cfg["max_pdv_ruta"])
        if len(pool) < n_total:
            log.warning("El universo disponible (%s) es menor que la muestra solicitada (%s); se tomarán los disponibles.", len(pool), n_total)
            n_total = len(pool)

        # Fijos obligatorios y Prioritarios ----------------------------------
        col_f = cfg.get("columna_fijo")
        if col_f in pool.columns:
            es_fijo = self._normalizar_tipo(pool[col_f]) == "FIJO"
        else:
            es_fijo = pool.get("Es_Fijo", pd.Series(False, index=pool.index))
            es_fijo = es_fijo.fillna(False).astype(bool)
        es_fijo |= pool["_FIJO_FORZADO"].fillna(False)
        fijos = pool[es_fijo].copy()

        codigos_prio = cfg.get("codigos_prioritarios", [])
        if codigos_prio:
            col_cod = cfg.get("llave_universo", "CÓDIGO")
            col_encontrada = next((c for c in [col_cod, "CÓDIGO", "CODIGO", "Codigo", "RefIDEmbotellador"] if c in pool.columns), None)
            if col_encontrada:
                s_cod = normalizar_llave(pool[col_encontrada])
                set_prio = {str(x).strip() for x in codigos_prio}
                es_prio = s_cod.isin(set_prio)
                prioritarios = pool[es_prio].copy()
                log.info("Titulares prioritarios forzados (%s): %s", len(set_prio), len(prioritarios))
                fijos = pd.concat([fijos, prioritarios]).drop_duplicates()
        if len(fijos) > n_total:
            n_total = len(fijos)
        log.info("Titulares fijos obligatorios: %s", f"{len(fijos):,}")
        resto = pool[~pool.index.isin(fijos.index)]

        # Nivel 1: selección de rutas ----------------------------------------
        rutas_sel = self._elegir_rutas(resto, fijos, n_total)

        # Cupos por ruta dentro de [mín, máx] --------------------------------
        cupos_ruta = self._cupos_por_ruta(rutas_sel, n_total - len(fijos))

        # Nivel 2: llenado por GEC -------------------------------------------
        cuotas_raw = {str(k).upper(): float(v) for k, v in cfg["cuotas_gec"].items()}
        if nueva_regla:
            objetivo_gec = {k: int(v * 4 if v >= 1 else round(tamano_base * 4 * v)) for k, v in cuotas_raw.items()}
        else:
            objetivo_gec = self._repartir(n_total, cuotas_raw)
        for gec, k in fijos["_GEC"].value_counts().items():
            objetivo_gec[gec] = max(0, objetivo_gec.get(gec, 0) - int(k))
        log.info("Cuota restante por GEC (tras fijos): %s", objetivo_gec)

        adicionales = self._llenar_rutas(resto, cupos_ruta, objetivo_gec)
        titulares = pd.concat([fijos, adicionales])

        conteo = titulares["_RUTA"].value_counts()
        log.info(
            "Titulares: %s en %s rutas | PDV/ruta mín=%s prom=%.1f máx=%s.",
            f"{len(titulares):,}", len(conteo), int(conteo.min()),
            conteo.mean(), int(conteo.max()),
        )
        return titulares

    def _seleccionar_titulares_proporcion(self, pool: pd.DataFrame) -> pd.DataFrame:
        """Apunta a PP=4 por GEC y canal; PP=6 es el techo de compensación.

        Los fijos cargados siempre entran. Se completan primero sus rutas y
        luego se abren rutas compactas de 10+ puntos, respetando su máximo.
        """
        cfg = self.cfg
        base_total = int(cfg.get("tamano_muestra", 0))
        cuotas_gec = self._cuotas_base(cfg.get("cuotas_gec", {}), base_total)
        cuotas_canal = {
            self._normalizar_canal(pd.Series([k])).iloc[0]: v
            for k, v in self._cuotas_base(cfg.get("cuotas_canal", {}), base_total).items()
        }
        if not cuotas_gec:
            raise ErrorSeleccion("Configure las cuotas GEC antes de seleccionar la muestra.")
        tiene_canal = bool(cfg.get("columna_canal") in pool.columns and cuotas_canal)
        disponibles_gec = Counter(pool["_GEC"])
        disponibles_canal = Counter(pool["_CANAL"])
        ideal_gec = {k: min(4 * v, disponibles_gec[k]) for k, v in cuotas_gec.items()}
        ideal_canal = {k: min(4 * v, disponibles_canal[k]) for k, v in cuotas_canal.items()} if tiene_canal else {}
        max_gec = {k: 6 * v for k, v in cuotas_gec.items()}
        max_canal = {k: 6 * v for k, v in cuotas_canal.items()} if tiene_canal else {}
        base_total = base_total or sum(cuotas_gec.values())

        forzados = pool["_FIJO_FORZADO"].fillna(False) | pool["_TIPO"].eq("FIJO")
        codigos_prio = cfg.get("codigos_prioritarios") or []
        if codigos_prio:
            col_codigo = next((c for c in (cfg.get("llave_universo"), "CÓDIGO", "CODIGO", "Codigo", "RefIDEmbotellador") if c in pool.columns), None)
            if col_codigo:
                forzados |= normalizar_llave(pool[col_codigo]).isin({str(c).strip() for c in codigos_prio})
        elegidos = set(pool.index[forzados])
        conteo_gec = Counter(pool.loc[list(elegidos), "_GEC"])
        conteo_canal = Counter(pool.loc[list(elegidos), "_CANAL"])
        conteo_ruta = Counter(pool.loc[list(elegidos), "_RUTA"])
        max_total = max(6 * base_total, len(elegidos))
        minimo_ruta = 10
        maximo_ruta = max(minimo_ruta, int(cfg.get("max_pdv_ruta", 10)))
        rutas: dict[str, pd.DataFrame] = {}
        dispersion_ruta: dict[str, float] = {}
        for ruta, sub in pool.groupby("_RUTA", sort=True, dropna=False):
            # Mantener solo las columnas usadas por el heurístico evita
            # duplicar universos anchos en el servidor gratuito.
            sub = sub[["_GEC", "_CANAL", "_LAT", "_LON"]].copy()
            puntos_validos = sub.loc[sub["_LAT"].notna() & sub["_LON"].notna()]
            if puntos_validos.empty:
                sub["_DIST_ANCLA"] = np.inf
                dispersion_ruta[ruta] = np.inf
            else:
                fijos_validos = puntos_validos.loc[puntos_validos.index.isin(elegidos)]
                if not fijos_validos.empty:
                    anclas = fijos_validos[["_LAT", "_LON"]].to_numpy(float)
                    distancias = cKDTree(anclas).query(puntos_validos[["_LAT", "_LON"]].to_numpy(float))[0]
                    sub["_DIST_ANCLA"] = pd.Series(distancias, index=puntos_validos.index).reindex(sub.index).fillna(np.inf)
                else:
                    lat_centro = float(puntos_validos["_LAT"].median())
                    lon_centro = float(puntos_validos["_LON"].median())
                    distancias = haversine_km(
                        puntos_validos["_LAT"].to_numpy(float), puntos_validos["_LON"].to_numpy(float),
                        np.float64(lat_centro), np.float64(lon_centro),
                    )
                    sub["_DIST_ANCLA"] = pd.Series(distancias, index=puntos_validos.index).reindex(sub.index).fillna(np.inf)
                dispersion_ruta[ruta] = float(sub["_DIST_ANCLA"].replace(np.inf, np.nan).median())
            rutas[ruta] = sub

        def faltan_cuotas() -> bool:
            return any(conteo_gec[k] < objetivo for k, objetivo in ideal_gec.items()) or (
                tiene_canal and any(conteo_canal[k] < objetivo for k, objetivo in ideal_canal.items())
            )

        def mejor(ruta: str, locales: set, gec_local: Counter, canal_local: Counter, completar_minimo: bool):
            opciones = []
            for indice, fila in rutas[ruta].iterrows():
                if indice in locales:
                    continue
                gec, canal = fila["_GEC"], fila["_CANAL"]
                if gec not in max_gec or gec_local[gec] >= max_gec[gec]:
                    continue
                if tiene_canal and (canal not in max_canal or canal_local[canal] >= max_canal[canal]):
                    continue
                falta_gec = max(0, ideal_gec[gec] - gec_local[gec]) / max(1, ideal_gec[gec])
                falta_canal = max(0, ideal_canal[canal] - canal_local[canal]) / max(1, ideal_canal[canal]) if tiene_canal else 0
                if not completar_minimo and falta_gec == 0 and falta_canal == 0:
                    continue
                opciones.append(((int(falta_gec > 0) + int(falta_canal > 0), falta_gec + falta_canal,
                                  -float(fila["_DIST_ANCLA"])), indice))
            return max(opciones)[1] if opciones else None

        def agregar(indice) -> None:
            elegidos.add(indice)
            fila = pool.loc[indice]
            conteo_gec[fila["_GEC"]] += 1
            conteo_canal[fila["_CANAL"]] += 1
            conteo_ruta[fila["_RUTA"]] += 1

        # Rutas con fijos: se completa el mínimo si hay oferta y margen PP.
        for ruta in sorted(conteo_ruta, key=lambda r: str(r)):
            if len(rutas[ruta]) < minimo_ruta:
                continue
            while conteo_ruta[ruta] < minimo_ruta and len(elegidos) < max_total:
                indice = mejor(ruta, elegidos, conteo_gec, conteo_canal, True)
                if indice is None:
                    break
                agregar(indice)

        def llenar_activa(ruta: str) -> None:
            limite = max(maximo_ruta, int(forzados.loc[rutas[ruta].index].sum()))
            while conteo_ruta[ruta] < limite and len(elegidos) < max_total and faltan_cuotas():
                indice = mejor(ruta, elegidos, conteo_gec, conteo_canal, False)
                if indice is None:
                    break
                agregar(indice)

        for ruta in sorted(conteo_ruta, key=lambda r: (dispersion_ruta.get(r, np.inf), str(r))):
            llenar_activa(ruta)

        # Elegir primero rutas con puntos cercanos; nunca abrir una ruta que
        # pudiera quedar bajo 10 si hay otra ruta completa disponible.
        pendientes = sorted(
            (ruta for ruta in rutas if ruta not in conteo_ruta and len(rutas[ruta]) >= minimo_ruta),
            key=lambda r: (dispersion_ruta[r], str(r)),
        )
        for ruta in pendientes:
            if not faltan_cuotas() or len(elegidos) >= max_total:
                break
            locales = set(elegidos)
            gec_local, canal_local = conteo_gec.copy(), conteo_canal.copy()
            lote = []
            for _ in range(minimo_ruta):
                indice = mejor(ruta, locales, gec_local, canal_local, True)
                if indice is None:
                    break
                locales.add(indice)
                lote.append(indice)
                fila = pool.loc[indice]
                gec_local[fila["_GEC"]] += 1
                canal_local[fila["_CANAL"]] += 1
            if len(lote) < minimo_ruta or len(elegidos) + len(lote) > max_total:
                continue
            for indice in lote:
                agregar(indice)
            llenar_activa(ruta)

        # Solo si no queda ninguna ruta de 10 que ayude a las cuotas se usan
        # rutas cortas para acercarse a la proporción objetivo.
        if faltan_cuotas() and len(elegidos) < max_total:
            for ruta in sorted((r for r in rutas if r not in conteo_ruta), key=lambda r: (dispersion_ruta[r], str(r))):
                while conteo_ruta[ruta] < min(len(rutas[ruta]), maximo_ruta) and faltan_cuotas() and len(elegidos) < max_total:
                    indice = mejor(ruta, elegidos, conteo_gec, conteo_canal, False)
                    if indice is None:
                        break
                    agregar(indice)

        if not elegidos:
            raise ErrorSeleccion("No hay puntos que cumplan las cuotas GEC y canal configuradas.")
        seleccion = pool.loc[pool.index.isin(elegidos)].copy()
        self.log.info("Proporción 4: %s titulares, %s fijos, %s rutas.", len(seleccion), int(forzados.sum()), len(conteo_ruta))
        return seleccion

    def _elegir_rutas(
        self, resto: pd.DataFrame, fijos: pd.DataFrame, n_total: int
    ) -> pd.DataFrame:
        """
        Devuelve un DataFrame por ruta con: disponibles, fijos, estrato,
        límites [L, U] de cupo adicional. Selecciona rutas estratificadas por
        departamento (dispersión) con probabilidad proporcional al tamaño.
        """
        cfg, log = self.cfg, self.log
        minr, maxr = int(cfg["min_pdv_ruta"]), int(cfg["max_pdv_ruta"])
        prom = int(cfg.get("promedio_pdv_ruta", 10))
        col_geo = cfg.get("columna_estrato_geo", "")

        info = resto.groupby("_RUTA").agg(
            disponibles=("_RUTA", "size"),
            estrato=(col_geo, "first") if col_geo in resto.columns
            else ("_RUTA", "first"),
            peso_cluster=("_PUNTAJE_CLUSTER", "sum"),
        )
        info["n_fijos"] = fijos["_RUTA"].value_counts().reindex(info.index).fillna(0).astype(int)
        # rutas que solo existen por sus fijos
        solo_fijas = fijos[~fijos["_RUTA"].isin(info.index)]["_RUTA"].value_counts()
        for r, k in solo_fijas.items():
            info.loc[r] = {
                "disponibles": 0, "estrato": "(fijos)",
                "peso_cluster": 0.0, "n_fijos": int(k),
            }

        info["U"] = np.minimum(maxr - info["n_fijos"], info["disponibles"]).clip(lower=0)
        info["L"] = np.maximum(minr - info["n_fijos"], 0)
        info["L"] = np.minimum(info["L"], info["U"])     # nunca L > U

        forzadas = info[info["n_fijos"] > 0]
        candidatas = info[(info["n_fijos"] == 0) & (info["U"] >= minr)]

        n_rutas_obj = max(int(np.ceil(n_total / prom)), 1)
        faltan = max(n_rutas_obj - len(forzadas), 0)

        # Estratificado por departamento, PPS dentro del estrato -------------
        elegidas: list[str] = []
        if faltan and len(candidatas):
            reparto = self._repartir(
                faltan,
                (candidatas.groupby("estrato")["peso_cluster"].sum()
                 / candidatas["peso_cluster"].sum()).to_dict(),
            )
            for estrato, k in reparto.items():
                sub = candidatas[candidatas["estrato"] == estrato]
                k = min(int(k), len(sub))
                if k <= 0:
                    continue
                pesos = sub["peso_cluster"].clip(lower=0)
                p = pesos / pesos.sum() if pesos.sum() > 0 else pd.Series(1 / len(sub), index=sub.index)
                elegidas += list(self.rng.choice(sub.index, size=k,
                                                 replace=False, p=p.to_numpy()))
        sel = info.loc[list(forzadas.index) + elegidas].copy()

        # Factibilidad: sum(L) <= n_adicional <= sum(U) ----------------------
        n_ad = n_total - int(info["n_fijos"].sum())
        pool_extra = candidatas.drop(index=[e for e in elegidas], errors="ignore")
        pool_extra = pool_extra.sort_values("disponibles", ascending=False)
        while sel["U"].sum() < n_ad and len(pool_extra):
            r = pool_extra.index[0]
            sel.loc[r] = info.loc[r]
            pool_extra = pool_extra.drop(index=r)
        while sel["L"].sum() > n_ad:
            quitables = sel[(sel["n_fijos"] == 0)].sort_values("disponibles")
            if quitables.empty:
                break
            sel = sel.drop(index=quitables.index[0])
        if sel["U"].sum() < n_ad:
            raise ErrorSeleccion(
                "No hay capacidad suficiente en las rutas para la muestra "
                "solicitada; revise mín/máx por ruta o el tamaño."
            )
        log.info("Rutas seleccionadas: %s (%s forzadas por fijos, objetivo %s).",
                 len(sel), len(forzadas), n_rutas_obj)
        return sel

    def _cupos_por_ruta(self, rutas: pd.DataFrame, n_adicional: int) -> pd.Series:
        """Cupos adicionales por ruta dentro de [L, U] que suman n_adicional."""
        q = rutas["L"].astype(int).copy()
        restante = n_adicional - int(q.sum())
        margen = (rutas["U"] - q).astype(int)
        while restante > 0 and margen.sum() > 0:
            peso = margen / margen.sum()
            exacto = peso * restante
            extra = np.minimum(np.floor(exacto).astype(int), margen)
            if extra.sum() == 0:      # resto mayor
                orden = (exacto - np.floor(exacto)).sort_values(ascending=False)
                for r in orden.index:
                    if restante == 0:
                        break
                    if margen[r] > 0:
                        q[r] += 1
                        margen[r] -= 1
                        restante -= 1
                continue
            q += extra
            margen -= extra
            restante -= int(extra.sum())
        return q[q + rutas["n_fijos"] > 0]

    def _llenar_rutas(
        self, resto: pd.DataFrame, cupos_ruta: pd.Series, objetivo_gec: dict
    ) -> pd.DataFrame:
        """
        Llena cada ruta asignando sus cupos al GEC con mayor necesidad
        relativa global que la ruta pueda abastecer; dentro del GEC toma los
        PDV de mayor Cupo_Restante (desempate aleatorio).
        """
        necesidad = {k: int(v) for k, v in objetivo_gec.items()}
        oferta = resto.groupby(["_RUTA", "_GEC"]).size()
        tomados: list = []

        rutas_orden = list(cupos_ruta.index)
        self.rng.shuffle(rutas_orden)
        for ruta in rutas_orden:
            cupo = int(cupos_ruta[ruta])
            if cupo <= 0:
                continue
            disponibles = {
                g: int(oferta.get((ruta, g), 0)) for g in necesidad
            }
            plan: dict[str, int] = {g: 0 for g in necesidad}
            for _ in range(cupo):
                candidatos = {
                    g: necesidad[g] for g in necesidad
                    if disponibles[g] - plan[g] > 0 and necesidad[g] > 0
                }
                if not candidatos:      # sin necesidad: usar lo que haya
                    candidatos = {
                        g: disponibles[g] - plan[g] for g in necesidad
                        if disponibles[g] - plan[g] > 0
                    }
                    if not candidatos:
                        break
                g = max(candidatos, key=candidatos.get)
                plan[g] += 1
                if necesidad.get(g, 0) > 0:
                    necesidad[g] -= 1
            sub = resto[resto["_RUTA"] == ruta].copy()
            sub["_azar"] = self.rng.random(len(sub))
            for g, k in plan.items():
                if k:
                    tomados.append(
                        sub[sub["_GEC"] == g]
                        .sort_values(["Cupo_Restante", "_PUNTAJE_CLUSTER", "_azar"],
                                     ascending=[False, False, False]).head(k)
                    )
        sobra = {g: k for g, k in necesidad.items() if k > 0}
        if sobra:
            self.log.warning(
                "Cuotas GEC con déficit tras el llenado por rutas: %s "
                "(cubierto con otros GEC dentro de la tolerancia).", sobra,
            )
        out = pd.concat(tomados).drop(columns="_azar") if tomados else resto.iloc[:0].copy()
        return out

    @staticmethod
    def _repartir(n: int, proporciones: dict) -> dict:
        """Reparto entero por resto mayor según proporciones."""
        total = sum(proporciones.values()) or 1
        exacto = {k: n * v / total for k, v in proporciones.items()}
        base = {k: int(v) for k, v in exacto.items()}
        faltan = n - sum(base.values())
        for k in sorted(exacto, key=lambda x: exacto[x] - base[x], reverse=True):
            if faltan == 0:
                break
            base[k] += 1
            faltan -= 1
        return base

    # ----------------------------------------------------------- dispersión --
    def _calcular_dispersion(self, t: pd.DataFrame) -> pd.DataFrame:
        """Distancia de cada titular al titular más cercano (km)."""
        t = t.copy()
        t["Dist_T_Cercano_km"] = np.nan
        con_gps = t["_LAT"].notna() & t["_LON"].notna()
        validos = t.loc[con_gps]
        if len(validos) >= 2:
            lat = validos["_LAT"].to_numpy(float)
            lon = validos["_LON"].to_numpy(float)
            # La cuerda de una esfera y la distancia haversine tienen el mismo
            # orden. El árbol encuentra el vecino sin crear una matriz N×N.
            lat_rad = np.radians(lat)
            lon_rad = np.radians(lon)
            puntos = np.column_stack((
                np.cos(lat_rad) * np.cos(lon_rad),
                np.cos(lat_rad) * np.sin(lon_rad),
                np.sin(lat_rad),
            ))
            vecinos = cKDTree(puntos).query(puntos, k=2)[1]
            indices = np.arange(len(validos))
            cercanos = np.where(vecinos[:, 0] == indices, vecinos[:, 1], vecinos[:, 0])
            distancias = haversine_km(lat, lon, lat[cercanos], lon[cercanos])
            t.loc[con_gps, "Dist_T_Cercano_km"] = np.round(distancias, 3)
        elif len(validos) == 1:
            t.loc[con_gps, "Dist_T_Cercano_km"] = np.inf
        self.log.info(
            "Dispersión: distancia al T más cercano — mín %.3f km, "
            "mediana %.3f km, promedio %.3f km.",
            t["Dist_T_Cercano_km"].min(), t["Dist_T_Cercano_km"].median(),
            t["Dist_T_Cercano_km"].mean(),
        )
        return t

    # ------------------------------------------------------------ suplentes --
    def _asignar_suplentes(self, t: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
        """
        N suplentes por titular (mismo GEC; prioriza misma ruta y cercanía).
        Cada suplente se asigna a un único titular; nivel S1..Sn por distancia.
        """
        cfg, log = self.cfg, self.log
        n_sup = int(cfg["ratio_suplentes"])
        if n_sup <= 0:
            return pd.DataFrame()
        
        # Buscar la columna del código dinámicamente
        col_cod = next((c for c in [cfg.get("llave_universo"), "CÓDIGO", "CODIGO", "Codigo", "RefIDEmbotellador", "RefID"] if c and c in pool.columns), None)
        if not col_cod:
            col_cod = pool.columns[0] # Fallback a la primera columna
            
        disponibles = pool.drop(index=t.index)          # elegibles no titulares
        usados: set = set()
        registros = []

        por_gec = {g: sub for g, sub in disponibles.groupby("_GEC")}
        for idx, fila in t.iterrows():
            if pd.isna(fila["_LAT"]) or pd.isna(fila["_LON"]):
                continue
            cand = por_gec.get(fila["_GEC"])
            if cand is None or cand.empty:
                continue
            cand = cand[~cand.index.isin(usados)]
            if cand.empty:
                continue
            d = haversine_km(
                np.float64(fila["_LAT"]), np.float64(fila["_LON"]),
                cand["_LAT"].to_numpy(float), cand["_LON"].to_numpy(float),
            )
            misma_ruta = (cand["_RUTA"] == fila["_RUTA"]).to_numpy()
            orden = np.lexsort((d, ~misma_ruta))        # 1º misma ruta, 2º cercanía
            elegidos = cand.index.to_numpy()[orden][:n_sup]
            dist_eleg = d[orden][:n_sup]
            for nivel, (i_sup, dk) in enumerate(zip(elegidos, dist_eleg), start=1):
                usados.add(i_sup)
                reg = disponibles.loc[i_sup].copy()
                reg["Nivel_Suplente"] = f"S{nivel}"
                reg["Titular_CODIGO"] = fila[col_cod]
                reg["Distancia_al_Titular_km"] = round(float(dk), 3)
                registros.append(reg)

        sup = pd.DataFrame(registros)
        completos = int((sup.groupby("Titular_CODIGO").size() == n_sup).sum()) if len(sup) else 0
        log.info("Suplentes asignados: %s (relación 1:%s; titulares con set "
                 "completo: %s de %s).",
                 f"{len(sup):,}", n_sup, f"{completos:,}", f"{len(t):,}")
        return sup

    # ----------------------------------------------------------- presentación
    def _presentar(self, df: pd.DataFrame, es_titular: bool) -> pd.DataFrame:
        """Limpia columnas técnicas y expone coordenadas normalizadas."""
        if df.empty:
            return df
        out = df.copy()
        out["Latitud_Normalizada"] = out.pop("_LAT")
        out["Longitud_Normalizada"] = out.pop("_LON")
        out = out.drop(columns=[
            c for c in ("_GEC", "_RUTA", "_CANAL", "_TIPO", "_PUNTAJE_CLUSTER")
            if c in out.columns
        ])
        out.insert(0, "Tipo", "TITULAR" if es_titular else "SUPLENTE")
        out["seleccion"] = "T" if es_titular else ""
        return out.reset_index(drop=True)

    def _resumenes_seleccion(
        self, pool: pd.DataFrame, titulares: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Resume rutas y la proporción real seleccionada/base por segmento."""
        rutas = titulares.groupby("_RUTA", dropna=False).agg(
            **{"Puntos seleccionados": ("_RUTA", "size"),
               "Fijos seleccionados": ("_TIPO", lambda valores: int((valores == "FIJO").sum()))}
        ).reset_index(names="Ruta").sort_values("Ruta")
        disponibles = pool.groupby("_RUTA", dropna=False).size()
        rutas["Elegibles disponibles"] = rutas["Ruta"].map(disponibles).astype(int)
        rutas["Meta mínima"] = 10
        rutas["Estado"] = np.where(
            rutas["Puntos seleccionados"] >= 10, "Cumple",
            np.where(rutas["Elegibles disponibles"] < 10, "Sin suficientes elegibles", "Por completar"),
        )

        def tabla(columna: str, cuotas: dict) -> pd.DataFrame:
            base = self._cuotas_base(cuotas, int(self.cfg.get("tamano_muestra", 0)))
            if columna == "_CANAL":
                base = {self._normalizar_canal(pd.Series([clave])).iloc[0]: valor for clave, valor in base.items()}
            conteos = titulares[columna].value_counts()
            categorias = list(base) + [valor for valor in conteos.index if valor not in base]
            filas = []
            for categoria in categorias:
                cantidad = int(conteos.get(categoria, 0))
                muestra = base.get(categoria)
                filas.append({
                    "Etiquetas de fila": categoria,
                    "Cuenta de RefID": cantidad,
                    "M": muestra,
                    "PP": round(cantidad / muestra, 1) if muestra else None,
                })
            filas.append({
                "Etiquetas de fila": "Total general",
                "Cuenta de RefID": int(len(titulares)),
                "M": sum(base.values()),
                "PP": round(len(titulares) / sum(base.values()), 1) if sum(base.values()) else None,
            })
            return pd.DataFrame(filas)

        return rutas, tabla("_GEC", self.cfg.get("cuotas_gec", {})), tabla("_CANAL", self.cfg.get("cuotas_canal", {}))

    @staticmethod
    def _cuotas_base(cuotas: dict, tamano_base: int) -> dict[str, int]:
        """Convierte cuotas porcentuales (Ecuador) a cantidades base enteras."""
        valores = {str(k).strip().upper(): float(v) for k, v in cuotas.items()}
        if valores and all(0 <= valor <= 1 for valor in valores.values()):
            return SelectorMuestra._repartir(tamano_base, valores)
        return {clave: int(round(valor)) for clave, valor in valores.items()}

    def _metricas(self, uni, pool, res: ResultadoSeleccion, sin_gps: int) -> dict:
        cfg = self.cfg
        t = res.titulares
        gec_normalizado = (
            normalizar_llave(t[cfg["columna_gec"]])
            if len(t) else pd.Series(dtype="object")
        )
        conteos_gec = gec_normalizado.value_counts()
        columna_ruta = cfg.get("columna_ruta") or cfg.get("columna_agencia")
        rutas = t[columna_ruta].value_counts() if len(t) and columna_ruta in t else pd.Series(dtype=int)

        out_metrics = {
            "Universo elegible de entrada": len(uni),
            "Tiendas sin GPS válido": sin_gps,
            "Titulares (T) seleccionados": len(t),
            "Suplentes (S) asignados": len(res.suplentes),
            "Muestra GEC (titulares)": len(t),
            "Rutas utilizadas": int(len(rutas)),
            "PDV por ruta (mín/prom/máx)": (
                f"{int(rutas.min())}/{rutas.mean():.1f}/{int(rutas.max())}"
                if len(rutas) else "0/0/0"
            ),
            "Dispersión: dist. al T más cercano (prom km)": (
                round(float(t["Dist_T_Cercano_km"].mean()), 3) if len(t) else 0
            ),
            "Clusters geográficos representados": (
                int(t["CLUSTER"].nunique()) if "CLUSTER" in t.columns else 0
            ),
            "Densidad promedio de titulares": (
                round(float(t["DENSIDAD"].mean()), 1) if "DENSIDAD" in t.columns and len(t) else 0
            ),
            "Semilla aleatoria": str(cfg.get("semilla")),
            "Fecha y hora de ejecución": f"{res.inicio:%Y-%m-%d %H:%M:%S}",
            "Tiempo total del proceso": f"{res.duracion_seg:.1f} segundos",
        }
        orden_gec = ["ORO", "PLATA", "BRONCE"] + [
            valor for valor in conteos_gec.index
            if valor not in {"ORO", "PLATA", "BRONCE"}
        ]
        posicion = list(out_metrics).index("Rutas utilizadas")
        items = list(out_metrics.items())
        detalle_gec = []
        for gec in orden_gec:
            cantidad = int(conteos_gec.get(gec, 0))
            proporcion = (cantidad / len(t) * 100) if len(t) else 0.0
            detalle_gec.append((f"  - {gec}", f"{cantidad:,} ({proporcion:.1f}%)"))
        out_metrics = dict(items[:posicion] + detalle_gec + items[posicion:])
        return out_metrics
