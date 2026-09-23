"""Regresiones de fijos obligatorios y proporciones de la muestra web."""

import unittest
from pathlib import Path

import pandas as pd

from seleccion import SelectorMuestra


def selector_cr(tamano=2):
    config = {
        "pais_activo": "Costa Rica",
        "tamano_muestra": tamano,
        "cuotas_gec": {"ORO": tamano},
        "cuotas_canal": {"OFF": tamano},
        "columna_codigo": "Codigo",
        "llave_universo": "Codigo",
        "columna_gec": "GEC",
        "columna_canal": "Canal",
        "columna_ruta": "Ruta",
        "columna_fijo": "Fijo",
        "valor_fijo": "SI",
        "columna_lat": "Latitud",
        "columna_lon": "Longitud",
        "pxr_minimo": 1,
        "min_pdv_ruta": 10,
        "max_pdv_ruta": 12,
        "ratio_suplentes": 0,
    }
    selector = SelectorMuestra(Path("."), config)
    selector.pais_activo = "Costa Rica"
    return selector


def universo_cr(n=12):
    return pd.DataFrame({
        "Codigo": [f"P{i:02d}" for i in range(n)],
        "GEC": ["ORO"] * n,
        "Canal": ["OFF"] * n,
        "Ruta": ["R1"] * n,
        "Fijo": ["NO"] * n,
        "Latitud": [9.0 + i / 1000 for i in range(n)],
        "Longitud": [-84.0] * n,
        "ELEGIBLE": ["ELEGIBLE"] * n,
    })


class SeleccionProporcionTests(unittest.TestCase):
    def test_incluye_todo_fijo_cargado_aunque_no_elegible_sin_gps_o_fuera_del_universo(self):
        universo = universo_cr()
        universo.loc[0, "ELEGIBLE"] = "NO ELEGIBLE INC"
        externo = universo.iloc[[0]].copy()
        externo.loc[0, "Codigo"] = "EXTERNO"
        externo.loc[0, "Latitud"] = None
        externo.loc[0, "Longitud"] = None
        externo.loc[0, "ELEGIBLE"] = "NO ELEGIBLE INC"
        fijos = pd.concat([universo.iloc[[0]], externo], ignore_index=True)

        resultado = selector_cr().ejecutar(universo=universo, fijos=fijos)
        self.assertTrue({"P00", "EXTERNO"}.issubset(set(resultado.titulares["Codigo"])))
        seleccion = resultado.universo_revisado.set_index("Codigo")["seleccion"]
        self.assertEqual(seleccion["P00"], "T")
        self.assertEqual(seleccion["EXTERNO"], "T")
        self.assertEqual(resultado.metricas["Fijos sin GPS incluidos"], 1)
        self.assertGreaterEqual(resultado.resumen_rutas.loc[0, "Puntos seleccionados"], 10)

    def test_no_recorta_fijos_aunque_superen_la_muestra_base(self):
        universo = universo_cr()
        fijos = universo.iloc[:8].copy()
        resultado = selector_cr().ejecutar(universo=universo, fijos=fijos)
        self.assertEqual(set(fijos["Codigo"]) - set(resultado.titulares["Codigo"]), set())
        self.assertGreaterEqual(len(resultado.titulares), 10)

    def test_prioriza_vecinos_del_fijo_en_la_misma_ruta(self):
        universo = universo_cr()
        universo.loc[11, "Latitud"] = 10.5
        resultado = selector_cr().ejecutar(universo=universo, fijos=universo.iloc[[0]])
        self.assertEqual(len(resultado.titulares), 10)
        self.assertNotIn("P11", set(resultado.titulares["Codigo"]))
        self.assertEqual(set(resultado.titulares["Codigo"]), {f"P{i:02d}" for i in range(10)})

    def test_cuotas_porcentuales_ecuador_se_convierten_a_base(self):
        self.assertEqual(
            SelectorMuestra._cuotas_base(
                {"ORO": 0.5502, "PLATA": 0.2602, "BRONCE": 0.1896}, 807,
            ),
            {"ORO": 444, "PLATA": 210, "BRONCE": 153},
        )

    def test_ruta_llega_a_diez_y_exporta_resumen_pp(self):
        resultado = selector_cr().ejecutar(universo=universo_cr())
        self.assertEqual(len(resultado.titulares), 10)
        self.assertEqual(resultado.resumen_rutas.loc[0, "Estado"], "Cumple")
        self.assertEqual(resultado.resumen_gec.loc[0, "M"], 2)
        self.assertEqual(resultado.resumen_gec.loc[0, "PP"], 5.0)
        self.assertEqual(resultado.resumen_canal.loc[0, "PP"], 5.0)
        self.assertEqual(int((resultado.universo_revisado["seleccion"] == "T").sum()), 10)

    def test_gec_y_canal_apuntan_a_cuatro_sin_pasar_de_seis(self):
        universo = universo_cr(60)
        universo["GEC"] = ["ORO", "PLATA"] * 30
        universo["Canal"] = ["OFF", "ON", "ON", "OFF"] * 15
        universo["Ruta"] = [f"R{i // 12}" for i in range(60)]
        selector = selector_cr(10)
        selector.cfg["cuotas_gec"] = {"ORO": 5, "PLATA": 5}
        selector.cfg["cuotas_canal"] = {"OFF": 5, "ON": 5}
        resultado = selector.ejecutar(universo=universo)
        for resumen in (resultado.resumen_gec, resultado.resumen_canal):
            segmentos = resumen[resumen["Etiquetas de fila"] != "Total general"]
            self.assertTrue((segmentos["PP"] >= 4).all())
            self.assertTrue((segmentos["PP"] <= 6).all())
        self.assertTrue((resultado.resumen_rutas["Puntos seleccionados"] >= 10).all())

    def test_ecuador_funciona_con_agencia_y_cuotas_porcentuales(self):
        universo = universo_cr()
        universo["Agencia"] = universo.pop("Ruta")
        universo.loc[6:, "GEC"] = "PLATA"
        universo["Canal"] = ["OFF", "ON"] * 6
        selector = selector_cr(2)
        selector.pais_activo = "Ecuador"
        selector.cfg.update({
            "columna_ruta": "", "columna_agencia": "Agencia",
            "cuotas_gec": {"ORO": 0.5, "PLATA": 0.5},
            "cuotas_canal": {"OFF": 1, "ON": 1},
        })
        resultado = selector.ejecutar(universo=universo)
        self.assertEqual(len(resultado.titulares), 10)
        self.assertEqual(resultado.resumen_gec.loc[0, "M"], 1)
        self.assertEqual(resultado.resumen_gec.loc[1, "M"], 1)

    def test_republica_dominicana_tambien_respeta_minimo_y_fijos_cargados(self):
        universo = universo_cr()
        selector = selector_cr(10)
        selector.pais_activo = "República Dominicana"
        selector.cfg["max_pdv_ruta"] = 7
        fijos = universo.iloc[:2].copy()
        resultado = selector.ejecutar(universo=universo, fijos=fijos)
        self.assertEqual(len(resultado.titulares), 10)
        self.assertTrue(set(fijos["Codigo"]).issubset(set(resultado.titulares["Codigo"])))
        self.assertEqual(resultado.resumen_rutas.loc[0, "Estado"], "Cumple")


if __name__ == "__main__":
    unittest.main()
