"""Estado de archivos y parámetros compartidos por los módulos de la web.

Los Excel viven solo en la sesión web; no se guardan en GitHub ni en
la carpeta sincronizada del ejecutable.
"""

from __future__ import annotations

import copy
import hashlib
import io
import math
from datetime import date, datetime
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
from openpyxl import Workbook


def safe_workbook_name(name: str) -> str:
    clean = Path(PureWindowsPath(str(name)).name).name
    if clean in {"", ".", ".."} or Path(clean).suffix.lower() not in {".xlsx", ".xlsm"}:
        raise ValueError("Seleccione un archivo Excel .xlsx o .xlsm válido.")
    return clean


def workbook_record(name: str, content: bytes) -> dict:
    """Valida un Excel y guarda sus hojas junto con el contenido de la sesión."""
    clean = safe_workbook_name(name)
    if not content:
        raise ValueError(f"El archivo {clean} está vacío.")
    try:
        with pd.ExcelFile(io.BytesIO(content), engine="openpyxl") as workbook:
            sheets = tuple(workbook.sheet_names)
    except Exception as exc:
        raise ValueError(f"No se pudo abrir {clean} como Excel: {exc}") from exc
    if not sheets:
        raise ValueError(f"El archivo {clean} no contiene hojas.")
    return {
        "name": clean,
        "content": content,
        "sheets": sheets,
        "fingerprint": hashlib.sha256(content).hexdigest(),
    }


def workbook_columns(record: dict, sheet: str) -> list[str]:
    if sheet not in record["sheets"]:
        raise ValueError(f"La hoja '{sheet}' no existe en {record['name']}.")
    frame = pd.read_excel(io.BytesIO(record["content"]), sheet_name=sheet, nrows=0, engine="openpyxl")
    return [str(column).strip() for column in frame.columns]


def write_workbook(record: dict, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / safe_workbook_name(record["name"])
    target.write_bytes(record["content"])
    return target


def write_spreadsheet(path: Path, frames: dict[str, pd.DataFrame]) -> None:
    """Escribe resultados Excel por filas para evitar acumular celdas en RAM."""
    book = Workbook(write_only=True)
    for name, frame in frames.items():
        sheet = book.create_sheet(str(name)[:31])
        sheet.append([str(column) for column in frame.columns])
        for values in frame.itertuples(index=False, name=None):
            row = []
            for value in values:
                if value is None or value is pd.NA or value is pd.NaT:
                    value = None
                elif isinstance(value, (float, np.floating)) and not math.isfinite(value):
                    value = None if math.isnan(value) else str(value)
                elif isinstance(value, np.generic):
                    value = value.item()
                elif isinstance(value, pd.Timestamp):
                    value = value.to_pydatetime()
                elif not isinstance(value, (str, int, float, bool, date, datetime)):
                    value = str(value)
                row.append(value)
            sheet.append(row)
    book.save(path)


def selection_source_options(has_depuracion: bool, has_uploaded_file: bool) -> list[str]:
    """Ofrece solo fuentes realmente disponibles, priorizando la depuración."""
    options = []
    if has_depuracion:
        options.append("Resultado de depuración")
    if has_uploaded_file:
        options.append("Archivo cargado")
    return options


def parse_priority_codes(text: str) -> list[int | str]:
    result: list[int | str] = []
    for raw in text.split(","):
        value = raw.strip()
        if value:
            result.append(int(value) if value.isdigit() else value)
    return result


def synchronize_country_parameters(country_config: dict) -> None:
    """Replica las relaciones que aplica Guardar Configuración en escritorio."""
    dep = country_config["modulo_depuracion"]
    sel = country_config["modulo_seleccion"]

    for field in ("columna_lat", "columna_lon"):
        sel[field] = dep.get(field, "")
    for field in ("columna_gec", "columna_fijo", "valor_fijo"):
        dep[field] = sel.get(field, "")
    if "columna_canal" in sel:
        dep["columna_canal"] = sel["columna_canal"]

    minimum = int(dep.get("pxr_minimo", sel.get("pxr_minimo", 10)))
    if minimum <= 0:
        raise ValueError("El PXR mínimo debe ser mayor que cero.")
    dep["pxr_minimo"] = sel["pxr_minimo"] = minimum

    repeat = dep.get("rotacion", {}).get("cupos_por_gec", {})
    if repeat and any(int(value) <= 0 for value in repeat.values()):
        raise ValueError("Los límites REP deben ser mayores que cero.")

    cluster = sel.get("dbscan") or sel.get("cluster") or {"eps": 0.01, "min_samples": 1}
    sel["dbscan"] = copy.deepcopy(cluster)
    sel["cluster"] = copy.deepcopy(cluster)

    route_sample = sel.setdefault("muestra_ruta", {})
    gec = sel.get("cuotas_gec", {})
    # Ecuador usa proporciones de GEC; convertirlas a enteros destruiría su
    # configuración y el tamaño objetivo de la muestra.
    if gec and all(float(value) >= 1 for value in gec.values()):
        sel["tamano_muestra"] = sum(int(value) for value in gec.values())
        for name in ("ORO", "PLATA", "BRONCE"):
            if name in gec:
                route_sample[name.lower()] = int(gec[name])

    for name, route_key in (("FIJO", "fijos"), ("VARIABLE", "variables")):
        if name in sel.get("cuotas_tipo", {}):
            route_sample[route_key] = int(sel["cuotas_tipo"][name])
    channel = sel.get("cuotas_canal", {})
    for names, route_key in (
        (("ON", "ON PREMISE"), "canal_on"),
        (("OFF", "HOME MARKET TRADICIONAL"), "canal_off"),
    ):
        for name in names:
            if name in channel:
                route_sample[route_key] = int(channel[name])
                break
