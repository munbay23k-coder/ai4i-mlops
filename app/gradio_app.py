
import os

import gradio as gr
import requests


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def predict_failure(
    type,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear,
):
    data = {
        "type": type,
        "air_temperature": air_temperature,
        "process_temperature": process_temperature,
        "rotational_speed": int(rotational_speed),
        "torque": torque,
        "tool_wear": int(tool_wear),
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=data,
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()

        prediction = (
            "Falla detectada"
            if result["prediction"] == 1
            else "Sin falla detectada"
        )

        probabilities = {
            "Sin falla": result["probability_no_failure"],
            "Con falla": result["probability_failure"],
        }

        return prediction, probabilities

    except requests.RequestException as error:
        raise gr.Error(f"Error al consultar la API: {error}")


demo = gr.Interface(
    fn=predict_failure,
    inputs=[
        gr.Dropdown(
            choices=["L", "M", "H"],
            value="L",
            label="Tipo de producto",
        ),
        gr.Number(value=300.0, label="Temperatura del aire [K]"),
        gr.Number(value=310.0, label="Temperatura del proceso [K]"),
        gr.Number(value=1500, precision=0, label="Velocidad [rpm]"),
        gr.Number(value=40.0, label="Torque [Nm]"),
        gr.Number(value=100, precision=0, label="Desgaste [min]"),
    ],
    outputs=[
        gr.Textbox(label="Predicción"),
        gr.Label(label="Probabilidades estimadas"),
    ],
    title="AI4I - Mantenimiento predictivo",
    description="Clasificación de condiciones de falla utilizando HistGradientBoosting.",
)


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("PORT", "7860")),
    )
