import os
import json
import gc
import joblib
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import tensorflow as tf

from pathlib import Path
from flask import Flask, render_template, request


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODELOS_DIR = BASE_DIR / "modelos"


# Cargar scaler
scaler_modelo7 = joblib.load(
    MODELOS_DIR / "scaler_modelo7.pkl"
)

# Cargar columnas
with open(MODELOS_DIR / "columnas_modelo7.json", "r", encoding="utf-8") as archivo:
    columnas_modelo7 = json.load(archivo)

# Cargar métricas
with open(MODELOS_DIR / "metricas_modelos.json", "r", encoding="utf-8") as archivo:
    metricas_modelos = json.load(archivo)


modelos_archivos = {
    "Modelo 7 original reentrenado": "modelo7_original_reentrenado.h5",
    "Modelo 7.1": "modelo7_1.h5",
    "Modelo 7.3 final": "modelo7_3_final.h5"
}


def calcular_edad(fecha_nacimiento):
    fecha_nacimiento = pd.to_datetime(
        fecha_nacimiento,
        errors="coerce"
    )

    if pd.isna(fecha_nacimiento):
        return None

    hoy = pd.Timestamp.today().normalize()

    edad = (
        hoy.year
        - fecha_nacimiento.year
        - int(
            (hoy.month, hoy.day)
            < (fecha_nacimiento.month, fecha_nacimiento.day)
        )
    )

    return edad


def validar_edad_experiencia(fecha_nacimiento, experiencia):
    edad = calcular_edad(fecha_nacimiento)

    if edad is None:
        return False, None, "La fecha de nacimiento no es válida."

    if edad < 18:
        return False, edad, "La persona no cumple la edad mínima laboral."

    experiencia_maxima = edad - 18

    if experiencia > experiencia_maxima:
        return (
            False,
            edad,
            f"La experiencia no es compatible con la edad. Máximo permitido: {experiencia_maxima} años."
        )

    return True, edad, "Datos consistentes."


def preparar_entrada(experience, qualification, university, role, cert):
    entrada = pd.DataFrame(
        np.zeros((1, len(columnas_modelo7))),
        columns=columnas_modelo7
    )

    if "Experience" in entrada.columns:
        entrada.loc[0, "Experience"] = experience

    columnas_posibles = [
        f"Qualification_{qualification}",
        f"University_{university}",
        f"Role_{role}",
        f"Cert_{cert}"
    ]

    for columna in columnas_posibles:
        if columna in entrada.columns:
            entrada.loc[0, columna] = 1

    entrada_norm = scaler_modelo7.transform(entrada)

    return entrada_norm.astype("float32")


def predecir_con_modelo(nombre_modelo, entrada_norm):
    archivo_modelo = modelos_archivos[nombre_modelo]
    ruta_modelo = MODELOS_DIR / archivo_modelo

    modelo = tf.keras.models.load_model(
        ruta_modelo,
        compile=False
    )

    prediccion = modelo(
        entrada_norm,
        training=False
    ).numpy()[0][0]

    del modelo
    tf.keras.backend.clear_session()
    gc.collect()

    return float(prediccion)


@app.route("/", methods=["GET", "POST"])
def index():
    resultados = None
    error = None
    datos_usuario = None

    if request.method == "POST":
        try:
            nombre = request.form.get("nombre")
            fecha_nacimiento = request.form.get("fecha_nacimiento")
            phone_number = request.form.get("phone_number")

            experience = int(request.form.get("experience"))
            qualification = request.form.get("qualification")
            university = request.form.get("university")
            role = request.form.get("role")
            cert = request.form.get("cert")

            valido, edad, mensaje = validar_edad_experiencia(
                fecha_nacimiento,
                experience
            )

            datos_usuario = {
                "nombre": nombre,
                "fecha_nacimiento": fecha_nacimiento,
                "phone_number": phone_number,
                "edad": edad,
                "experience": experience,
                "qualification": qualification,
                "university": university,
                "role": role,
                "cert": cert
            }

            if not valido:
                error = mensaje
            else:
                entrada_norm = preparar_entrada(
                    experience,
                    qualification,
                    university,
                    role,
                    cert
                )

                resultados = []

                for nombre_modelo in modelos_archivos:
                    salario_predicho = predecir_con_modelo(
                        nombre_modelo,
                        entrada_norm
                    )

                    metricas = metricas_modelos.get(
                        nombre_modelo,
                        {}
                    )

                    resultados.append({
                        "modelo": nombre_modelo,
                        "salario_quincenal": salario_predicho,
                        "mae": metricas.get("mae", 0),
                        "rmse": metricas.get("rmse", 0),
                        "descripcion": metricas.get("descripcion", "")
                    })

                resultados = sorted(
                    resultados,
                    key=lambda item: item["mae"]
                )

        except Exception as e:
            error = f"Ocurrió un error: {e}"

    return render_template(
        "index.html",
        resultados=resultados,
        error=error,
        datos_usuario=datos_usuario
    )


if __name__ == "__main__":
    app.run(debug=False)
