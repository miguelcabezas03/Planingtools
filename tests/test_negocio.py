from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from depuracion import DepuradorUniverso, ResultadoDepuracion
from normalizacion import generar_codigo_puente, normalizar_llave
from seleccion import ResultadoSeleccion, SelectorMuestra, dbscan_coordenadas, haversine_km


class CodigoPuenteTests(unittest.TestCase):
    def test_costa_rica_prefijo_por_longitud(self):
        origen = pd.Series([900001431, 1002000506])
        generado = generar_codigo_puente(
            origen,
            n_digitos=7,
            prefijos_por_longitud={"9": "150"},
            prefijo_defecto="154",
        )
        self.assertEqual(generado.tolist(), ["1500001431", "1542000506"])
        self.assertEqual(origen.tolist(), [900001431, 1002000506])

    def test_nicaragua_prefijo_constante(self):
        generado = generar_codigo_puente(
            pd.Series([1002000506]), prefijo="160", n_digitos=7,
        )
        self.assertEqual(generado.tolist(), ["1602000506"])


class ElegibilidadTests(unittest.TestCase):
    def setUp(self):
        cfg = {
            "rotacion": {"cupos_por_gec": {"ORO": 6, "PLATA": 4, "BRONCE": 2}}
        }
        self.depurador = DepuradorUniverso(Path("."), cfg)

    def test_prioridad_y_limites_rep(self):
        datos = pd.DataFrame({
            "_ES_FIJO": [False, False, False, True],
            "_DENTRO_POLIGONO_PAIS": [True, True, False, False],
            "_DENTRO_DELIMITACION_MUESTRA": [True, True, True, False],
            "ENV_Total": [0, 0, 0, 0],
            "NC_Actual": [0, 0, 0, 0],
            "I_Total": [0, 0, 0, 0],
            "Estatus": ["Sin Visita"] * 4,
            "Comentario": ["Elegible"] * 4,
            "Gec": ["ORO", "PLATA", "BRONCE", "ORO"],
            "A_Actual": [6, 3, 9, 99],
        })
        resultado = self.depurador._evaluar_elegibilidad(datos, "Gec")
        self.assertEqual(resultado.tolist(), [
            "NO ELEGIBLE REP", "ELEGIBLE", "NO ELEGIBLE EG", "ELEGIBLE",
        ])

    def test_limites_rep_configurados_controlan_no_elegible_rep(self):
        datos = pd.DataFrame({
            "_ES_FIJO": [False] * 6,
            "_DENTRO_POLIGONO_PAIS": [True] * 6,
            "_DENTRO_DELIMITACION_MUESTRA": [True] * 6,
            "ENV_Total": [0] * 6,
            "NC_Actual": [0] * 6,
            "I_Total": [0] * 6,
            "Estatus": ["Sin Visita"] * 6,
            "Comentario": ["Elegible"] * 6,
            "Gec": ["ORO", "ORO", "PLATA", "PLATA", "BRONCE", "BRONCE"],
            "A_Actual": [5, 6, 3, 4, 1, 2],
        })
        resultado = self.depurador._evaluar_elegibilidad(datos, "Gec")
        self.assertEqual(
            resultado.tolist(),
            [
                "ELEGIBLE", "NO ELEGIBLE REP",
                "ELEGIBLE", "NO ELEGIBLE REP",
                "ELEGIBLE", "NO ELEGIBLE REP",
            ],
        )

    def test_rep_acepta_mayusculas_espacios_sufijo_y_numeros_con_coma(self):
        datos = pd.DataFrame({
            "_ES_FIJO": [False] * 6,
            "_DENTRO_POLIGONO_PAIS": [True] * 6,
            "_DENTRO_DELIMITACION_MUESTRA": [True] * 6,
            "ENV_Total": [0] * 6,
            "NC_Actual": [0] * 6,
            "I_Total": [0] * 6,
            "Estatus": ["Sin Visita"] * 6,
            "Comentario": ["Elegible"] * 6,
            "Gec": ["oro", "ORo", " Oro REP ", "plata", "PLATA REP", "bronce"],
            "A_Actual": ["6,0", 6, 6, "4,0", 4, "2,0"],
        })
        resultado = self.depurador._evaluar_elegibilidad(datos, "Gec")
        self.assertEqual(resultado.tolist(), ["NO ELEGIBLE REP"] * 6)

    def test_cruce_conserva_codigo_original(self):
        cfg = {
            "columna_puente": "Codigo D&N",
            "llave_incidencias": "Id_PDV",
            "n_digitos_cruce": 7,
            "prefijos_codigo_puente_por_longitud": {"9": "150"},
            "prefijo_codigo_puente_defecto": "154",
        }
        depurador = DepuradorUniverso(Path("."), cfg)
        universo = pd.DataFrame({"RefIDEmbotellador": [900001431]})
        incidencias = pd.DataFrame({"Id_PDV": [1500001431]})
        incidencias["_LLAVE"] = normalizar_llave(incidencias["Id_PDV"])
        depurador._preparar_llaves_cruce(
            universo, incidencias, "RefIDEmbotellador",
        )
        self.assertEqual(universo.loc[0, "RefIDEmbotellador"], 900001431)
        self.assertEqual(universo.loc[0, "Codigo D&N"], "1500001431")


class ReglaPXRTests(unittest.TestCase):
    def test_depuracion_aplica_revision_y_pxr_antes_de_exportar(self):
        config_global = {
            "pais_activo": "Prueba",
            "paises": {
                "Prueba": {
                    "modulo_seleccion": {
                        "columna_ruta": "Ruta",
                        "pxr_minimo": 99,
                    },
                },
            },
        }
        depurador = DepuradorUniverso(
            Path("."),
            {"columna_puente": "Codigo D&N", "pxr_minimo": 3},
            config_global,
        )
        revisados = pd.DataFrame({
            "Codigo D&N": ["1", "2", "3", "4", "5"],
            "Ruta": ["A", "A", "B", "B", "B"],
            "ELEGIBLE": [
                "ELEGIBLE", "ELEGIBLE", "NO ELEGIBLE EG", "ELEGIBLE", "ELEGIBLE",
            ],
        })
        resultado = depurador.aplicar_revision_geografica(
            ResultadoDepuracion(metricas={}), revisados,
        )
        self.assertEqual(resultado.elegibles["PXR"].tolist(), [2, 2, 0, 2, 2])
        self.assertEqual(
            resultado.elegibles["ELEGIBLE"].tolist(),
            [
                "NO ELEGIBLE PXR <3", "NO ELEGIBLE PXR <3",
                "NO ELEGIBLE EG", "NO ELEGIBLE PXR <3", "NO ELEGIBLE PXR <3",
            ],
        )
        self.assertEqual(resultado.metricas["PXR mínimo configurado"], 3)
        self.assertEqual(
            resultado.metricas["Excluidas por PXR (NO ELEGIBLE PXR <3)"], 4,
        )
        self.assertEqual(resultado.metricas["Tiendas elegibles finales"], 0)

    def test_pxr_cuenta_codigo_por_ruta_y_respeta_exclusiones_manuales(self):
        selector = SelectorMuestra(Path("."), {
            "columna_ruta": "Ruta",
            "columna_codigo": "Codigo D&N",
            "pxr_minimo": 3,
        })
        universo = pd.DataFrame({
            "Codigo D&N": ["1", "2", "3", "4", "5"],
            "Ruta": ["A", "A", "B", "B", "B"],
            "ELEGIBLE": [
                "ELEGIBLE", "ELEGIBLE", "NO ELEGIBLE EG", "ELEGIBLE", "ELEGIBLE",
            ],
        })
        resultado = selector._aplicar_regla_pxr(universo)
        self.assertEqual(resultado["PXR"].tolist(), [2, 2, 0, 2, 2])
        self.assertEqual(
            resultado["ELEGIBLE"].tolist(),
            [
                "NO ELEGIBLE PXR <3", "NO ELEGIBLE PXR <3",
                "NO ELEGIBLE EG", "NO ELEGIBLE PXR <3", "NO ELEGIBLE PXR <3",
            ],
        )

    def test_resumen_gec_incluye_conteo_y_proporcion_sin_campos_retirados(self):
        selector = SelectorMuestra(Path("."), {
            "columna_gec": "GEC",
            "columna_ruta": "Ruta",
        })
        titulares = pd.DataFrame({
            "GEC": ["ORO", "PLATA", "PLATA", "BRONCE"],
            "Ruta": ["A", "A", "B", "B"],
            "Dist_T_Cercano_km": [1.0, 2.0, 3.0, 4.0],
        })
        res = ResultadoSeleccion(titulares=titulares)
        metricas = selector._metricas(titulares, titulares, res, 0)
        self.assertEqual(metricas["  - ORO"], "1 (25.0%)")
        self.assertEqual(metricas["  - PLATA"], "2 (50.0%)")
        self.assertEqual(metricas["  - BRONCE"], "1 (25.0%)")
        self.assertNotIn("Relación T:S configurada", metricas)
        self.assertFalse(any("Rutas con <" in clave for clave in metricas))

    def test_seleccion_solo_recibe_registros_elegibles(self):
        selector = SelectorMuestra(Path("."), {
            "pais_activo": "Chile",
            "columna_codigo": "Codigo",
            "columna_gec": "GEC",
            "columna_ruta": "Ruta",
            "columna_lat": "Latitud",
            "columna_lon": "Longitud",
            "pxr_minimo": 1,
            "tamano_muestra": 1,
            "ratio_suplentes": 0,
        })
        selector.pais_activo = "Chile"
        universo = pd.DataFrame({
            "Codigo": ["A", "B"],
            "GEC": ["ORO", "ORO"],
            "Ruta": ["R1", "R1"],
            "Latitud": [9.9, 9.91],
            "Longitud": [-84.1, -84.11],
            "ELEGIBLE": ["ELEGIBLE", "NO ELEGIBLE EG"],
        })
        observados = []

        def seleccionar(pool):
            observados.extend(pool["Codigo"].tolist())
            return pool.head(1).copy()

        selector._seleccionar_titulares = seleccionar
        selector._calcular_dispersion = lambda df: df.assign(Dist_T_Cercano_km=0.0)
        selector._asignar_suplentes = lambda titulares, pool: pd.DataFrame()
        resultado = selector.ejecutar(universo=universo)
        self.assertEqual(observados, ["A"])
        self.assertEqual(resultado.titulares["Codigo"].tolist(), ["A"])
        self.assertEqual(selector.cfg["pais_activo"], "Chile")


class ClusterizacionSeleccionTests(unittest.TestCase):
    def test_dispersion_coincide_con_distancias_haversine(self):
        lat = np.array([8.98, 8.98, 9.1, -12.1, 0.0, 0.0])
        lon = np.array([-79.52, -79.52, -79.4, -77.0, 179.9, -179.9])
        titulares = pd.DataFrame({"_LAT": lat, "_LON": lon})
        resultado = SelectorMuestra(Path("."), {})._calcular_dispersion(titulares)
        esperado = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
        np.fill_diagonal(esperado, np.inf)
        np.testing.assert_array_equal(
            resultado["Dist_T_Cercano_km"].to_numpy(), np.round(esperado.min(axis=1), 3)
        )

    def test_dispersion_admite_miles_de_titulares_sin_matriz_cuadratica(self):
        titulares = pd.DataFrame({"_LAT": np.zeros(5000), "_LON": np.zeros(5000)})
        resultado = SelectorMuestra(Path("."), {})._calcular_dispersion(titulares)
        self.assertTrue((resultado["Dist_T_Cercano_km"] == 0).all())

    def test_dispersion_con_un_solo_titular(self):
        punto = pd.DataFrame({"_LAT": [0.0], "_LON": [0.0]})
        resultado = SelectorMuestra(Path("."), {})._calcular_dispersion(punto)
        self.assertTrue(np.isinf(resultado["Dist_T_Cercano_km"].iloc[0]))

    def test_dbscan_detecta_nucleo_y_ruido(self):
        etiquetas = dbscan_coordenadas(
            np.array([0.0, 0.001, 0.0, 10.0]),
            np.array([0.0, 0.0, 0.001, 10.0]),
            eps=0.01,
            min_samples=3,
        )
        self.assertEqual(etiquetas[:3].tolist(), [0, 0, 0])
        self.assertEqual(int(etiquetas[3]), -1)

    def test_optimizador_respeta_cuotas_y_prioriza_cluster_denso(self):
        selector = SelectorMuestra(Path("."), {
            "pais_activo": "ECUADOR",
            "tamano_muestra": 2,
            "cuotas_gec": {"ORO": 2},
            "cuotas_canal": {"OFF": 1, "ON": 1},
            "cuotas_tipo": {"FIJO": 1, "VARIABLE": 1},
            "cuotas_fijo_canal": {"FIJO_OFF": 1},
            "columna_gec": "GEC",
            "columna_ruta": "Ruta",
            "columna_canal": "CANAL",
            "columna_fijo": "FIJO",
            "min_pdv_ruta": 1,
            "max_pdv_ruta": 20,
            "dbscan": {"eps": 0.01, "min_samples": 2},
        })
        pool = pd.DataFrame({
            "Codigo": list("ABCDEF"),
            "GEC": ["ORO"] * 6,
            "Ruta": ["R1"] * 6,
            "CANAL": ["OFF", "ON", "OFF", "ON", "OFF", "ON"],
            "FIJO": ["FIJO", "VARIABLE", "VARIABLE", "FIJO", "FIJO", "VARIABLE"],
            "_GEC": ["ORO"] * 6,
            "_RUTA": ["R1"] * 6,
            "_LAT": [0.0, 0.001, 0.0, 0.001, 10.0, -10.0],
            "_LON": [0.0, 0.0, 0.001, 0.001, 10.0, -10.0],
        })
        preparado = selector._preparar_clusterizacion(pool)
        seleccionados = selector._seleccionar_titulares_ortools(preparado)
        self.assertIsNotNone(seleccionados)
        self.assertEqual(len(seleccionados), 2)
        self.assertEqual(set(seleccionados["Codigo"]), set("AB"))
        self.assertEqual(seleccionados["_CANAL"].value_counts().to_dict(), {"OFF": 1, "ON": 1})
        self.assertEqual(seleccionados["_TIPO"].value_counts().to_dict(), {"FIJO": 1, "VARIABLE": 1})
        self.assertTrue((seleccionados["DENSIDAD"] == 4).all())


if __name__ == "__main__":
    unittest.main()
