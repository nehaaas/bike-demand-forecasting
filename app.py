from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Bike Demand Forecast Review", layout="wide")
st.title("Bike Demand Forecast Review")
st.caption("Explore predictions on the held-out test period. This is an evaluation dashboard, not a live forecast.")

predictions_path = Path("reports/test_predictions.csv")
if not predictions_path.exists():
    st.warning(
        "Prediction results are not here yet. Run all cells in "
        "notebooks/bike_demand_forecasting.ipynb from the repository root first."
    )
    st.stop()

predictions = pd.read_csv(predictions_path)
predictions["datetime"] = pd.to_datetime(
    predictions["datetime"], dayfirst=True, errors="coerce"
)
if predictions["datetime"].isna().any():
    st.error("Some datetime values could not be parsed. Check reports/test_predictions.csv.")
    st.stop()
if predictions.empty:
    st.error("The prediction file is empty. Rerun the notebook to regenerate it.")
    st.stop()

available_days = sorted(predictions["datetime"].dt.date.unique())
selected_day = st.selectbox(
    "Choose a day from the test period",
    available_days,
    format_func=lambda value: value.strftime("%Y-%m-%d")
)
day = predictions[predictions["datetime"].dt.date == selected_day].copy()

mae = mean_absolute_error(day["actual_demand"], day["model_prediction"])
rmse = mean_squared_error(day["actual_demand"], day["model_prediction"]) ** 0.5
baseline_mae = mean_absolute_error(day["actual_demand"], day["weekly_baseline"])

left, middle, right = st.columns(3)
left.metric("Forecast MAE (bikes/hour)", f"{mae:.1f}")
middle.metric("Forecast RMSE", f"{rmse:.1f}")
right.metric("Weekly-baseline MAE", f"{baseline_mae:.1f}")

st.subheader(f"Actual and predicted demand — {selected_day:%Y-%m-%d}")
chart_data = day.set_index("datetime")[
    ["actual_demand", "model_prediction", "weekly_baseline"]
]
st.line_chart(chart_data)

st.subheader("Hourly results")
st.dataframe(
    day[["datetime", "actual_demand", "model_prediction", "weekly_baseline"]]
    .round(1),
    use_container_width=True,
    hide_index=True
)
st.caption(
    "Forecasts can be wrong, especially during unusual conditions. "
    "Use them as planning information, not as guaranteed demand."
)
