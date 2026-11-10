"""Gradio web application for late-delivery prediction."""

import gradio as gr

from predict import model, predict_order

# Read available categories from the trained model
preprocessor = model.named_steps["preprocessing"]

categorical_pipeline = preprocessor.named_transformers_["categorical"]

encoder = categorical_pipeline.named_steps["onehot"]

weather_choices = [str(x) for x in encoder.categories_[0]]
day_type_choices = [str(x) for x in encoder.categories_[1]]
vehicle_choices = [str(x) for x in encoder.categories_[2]]


def run_prediction(
    distance_km,
    items,
    hour,
    prep_minutes,
    courier_months,
    weather,
    day_type,
    vehicle,
):
    """Run the prediction and format the results."""

    result = predict_order(
        distance_km=distance_km,
        items=items,
        hour=hour,
        prep_minutes=prep_minutes,
        courier_months=courier_months,
        weather=weather,
        day_type=day_type,
        vehicle=vehicle,
    )

    return (
        result["late_percentage"],
        result["prediction"],
        result["voucher_decision"],
    )


with gr.Blocks(title="Late Delivery Predictor") as demo:
    gr.Markdown(
        """
        # Late Delivery Prediction System

        Predict the probability of a food delivery arriving late.

        Enter the order and courier details below.
        """
    )

    with gr.Row():
        distance = gr.Number(
            label="Delivery Distance (km)",
            value=5,
            minimum=0.01,
        )

        items = gr.Number(
            label="Number of Items",
            value=3,
            minimum=1,
            precision=0,
        )

        hour = gr.Number(
            label="Order Hour (0–23)",
            value=18,
            minimum=0,
            maximum=23,
            precision=0,
        )

    with gr.Row():
        prep_minutes = gr.Number(
            label="Preparation Time (minutes)",
            value=20,
            minimum=0,
            precision=0,
        )

        courier_months = gr.Number(
            label="Courier Experience (months)",
            value=12,
            minimum=0,
        )

    with gr.Row():
        weather = gr.Dropdown(
            choices=weather_choices,
            label="Weather",
            value=weather_choices[0],
        )

        day_type = gr.Dropdown(
            choices=day_type_choices,
            label="Day Type",
            value=day_type_choices[0],
        )

        vehicle = gr.Dropdown(
            choices=vehicle_choices,
            label="Vehicle",
            value=vehicle_choices[0],
        )

    predict_button = gr.Button(
        "Predict Delivery",
        variant="primary",
    )

    gr.Markdown("## Prediction Results")

    probability = gr.Number(
        label="Late Delivery Probability (%)",
        interactive=False,
    )

    prediction = gr.Textbox(
        label="Delivery Prediction",
        interactive=False,
    )

    voucher = gr.Textbox(
        label="Voucher Recommendation",
        interactive=False,
    )

    predict_button.click(
        fn=run_prediction,
        inputs=[
            distance,
            items,
            hour,
            prep_minutes,
            courier_months,
            weather,
            day_type,
            vehicle,
        ],
        outputs=[
            probability,
            prediction,
            voucher,
        ],
    )

    gr.Markdown(
        """
        ---
        **Business rule:** A voucher is recommended when the
        predicted late-delivery probability is at least 30%.

        Model: Logistic Regression
        """
    )


if __name__ == "__main__":
    demo.launch()
