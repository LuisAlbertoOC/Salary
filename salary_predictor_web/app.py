import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from flask import Flask, render_template, request


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODELOS_DIR = BASE_DIR / "modelos"


# Cargar modelos
modelos = {
    "Modelo 7 original reentrenado": tf.keras.models.load_model(
        MODELOS_DIR / "modelo7_original_reentrenado.h5",
        compile=False
    ),
    "Modelo 7.1": tf.keras.models.load_model(
        MODELOS_DIR / "modelo7_1.h5",
        compile=False
    ),
    "Modelo 7.3 final": tf.keras.models.load_model(
        MODELOS_DIR / "modelo7_3_final.h5",
        compile=False
    )
}

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
    """
    Crea una fila con las mismas columnas usadas durante el entrenamiento.
    """

    entrada = pd.DataFrame(
        np.zeros((1, len(columnas_modelo7))),
        columns=columnas_modelo7
    )

    if "Experience" in entrada.columns:
        entrada.loc[0, "Experience"] = experience

    columna_qualification = f"Qualification_{qualification}"
    columna_university = f"University_{university}"
    columna_role = f"Role_{role}"
    columna_cert = f"Cert_{cert}"

    for columna in [
        columna_qualification,
        columna_university,
        columna_role,
        columna_cert
    ]:
        if columna in entrada.columns:
            entrada.loc[0, columna] = 1

    entrada_norm = scaler_modelo7.transform(entrada)

    return entrada_norm


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

                for nombre_modelo, modelo in modelos.items():
                    prediccion = modelo.predict(
                        entrada_norm,
                        verbose=0
                    )[0][0]

                    metricas = metricas_modelos.get(
                        nombre_modelo,
                        {}
                    )

                    resultados.append({
                        "modelo": nombre_modelo,
                        "salario_quincenal": prediccion,
                        "mae": metricas.get("mae"),
                        "rmse": metricas.get("rmse"),
                        "descripcion": metricas.get("descripcion")
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
    app.run(debug=True)
