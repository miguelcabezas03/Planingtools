import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from server_config_store import load_configuration, save_configuration
from web_workflow import (
    parse_priority_codes, safe_workbook_name, synchronize_country_parameters,
    selection_source_options, workbook_columns, workbook_record, write_spreadsheet,
    write_workbook,
)


ROOT = Path(__file__).resolve().parents[1]


class WorkflowStateTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "Config" / "config.json").read_text(encoding="utf-8"))

    def test_workbook_is_validated_and_kept_by_name(self):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            pd.DataFrame({"Codigo": [1], "Lat": [2]}).to_excel(writer, sheet_name="Universo", index=False)
        record = workbook_record(r"C:\descargas\Base.xlsx", buffer.getvalue())
        self.assertEqual(record["name"], "Base.xlsx")
        self.assertEqual(record["sheets"], ("Universo",))
        self.assertEqual(workbook_columns(record, "Universo"), ["Codigo", "Lat"])
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(write_workbook(record, Path(temp)).read_bytes(), buffer.getvalue())
        with self.assertRaises(ValueError):
            safe_workbook_name("archivo.gpkg")

    def test_country_parameter_sync_preserves_ecuador_proportions(self):
        ecuador = copy.deepcopy(self.config["paises"]["Ecuador"])
        original = ecuador["modulo_seleccion"]["cuotas_gec"].copy()
        original_size = ecuador["modulo_seleccion"]["tamano_muestra"]
        synchronize_country_parameters(ecuador)
        self.assertEqual(ecuador["modulo_seleccion"]["cuotas_gec"], original)
        self.assertEqual(ecuador["modulo_seleccion"]["tamano_muestra"], original_size)

    def test_country_parameter_sync_mirrors_shared_fields(self):
        country = copy.deepcopy(self.config["paises"]["Chile"])
        dep = country["modulo_depuracion"]
        sel = country["modulo_seleccion"]
        dep["columna_lat"] = "NUEVA_LAT"
        sel["columna_gec"] = "NUEVO_GEC"
        sel["cuotas_gec"]["ORO"] = 2000
        synchronize_country_parameters(country)
        self.assertEqual(sel["columna_lat"], "NUEVA_LAT")
        self.assertEqual(dep["columna_gec"], "NUEVO_GEC")
        self.assertEqual(sel["tamano_muestra"], sum(sel["cuotas_gec"].values()))
        self.assertEqual(sel["muestra_ruta"]["oro"], 2000)
        self.assertEqual(sel["cluster"], sel["dbscan"])

    def test_priority_codes(self):
        self.assertEqual(parse_priority_codes("123, AB, 004, "), [123, "AB", 4])

    def test_selection_only_offers_available_sources(self):
        self.assertEqual(selection_source_options(False, False), [])
        self.assertEqual(selection_source_options(True, False), ["Resultado de depuración"])
        self.assertEqual(selection_source_options(False, True), ["Archivo cargado"])
        self.assertEqual(
            selection_source_options(True, True),
            ["Resultado de depuración", "Archivo cargado"],
        )

    def test_streamed_selection_workbook_preserves_data(self):
        data = pd.DataFrame({"Código": [1, 2], "GEC": ["ORO", "PLATA"], "GPS": [1.5, np.nan]})
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "Seleccion.xlsx"
            write_spreadsheet(output, {"Titulares": data, "Suplentes": data.iloc[0:0]})
            with pd.ExcelFile(output) as workbook:
                self.assertEqual(workbook.sheet_names, ["Titulares", "Suplentes"])
            pd.testing.assert_frame_equal(pd.read_excel(output, sheet_name="Titulares"), data)
            workbook = load_workbook(output, read_only=True)
            try:
                self.assertIsNone(workbook["Titulares"]["C3"].value)
            finally:
                workbook.close()

    def test_shared_server_save_reload_and_conflict(self):
        with tempfile.TemporaryDirectory() as temp:
            default = Path(temp) / "base.json"
            store = Path(temp) / "servidor.json"
            default.write_text(json.dumps(self.config), encoding="utf-8")
            config, revision = load_configuration(default, store)
            config["pais_activo"] = "Chile"
            saved_revision = save_configuration(config, revision, default, store)
            loaded, seen_revision = load_configuration(default, store)
            self.assertEqual(loaded["pais_activo"], "Chile")
            self.assertEqual(seen_revision, saved_revision)
            with self.assertRaises(ValueError):
                save_configuration(config, revision, default, store)


if __name__ == "__main__":
    unittest.main()
