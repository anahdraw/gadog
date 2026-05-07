from __future__ import annotations

import pickle
import math
import sys
import types
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st


MODEL_PATH = Path(__file__).with_name("treehouse.pkcls")
TARGET_LABEL = "Y house price of unit area"


def install_orange_palette_shim() -> None:
    """Avoid importing Orange's Qt widget stack for palette metadata in the pickle."""
    module_name = "Orange.widgets.utils.colorpalettes"
    if module_name in sys.modules:
        return

    colorpalettes = types.ModuleType(module_name)

    class ContinuousPalette:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self.state = state

    class Palette:
        pass

    class Flags(int):
        pass

    ContinuousPalette.__module__ = module_name
    Palette.__module__ = module_name
    Flags.__module__ = module_name
    Palette.Flags = Flags

    colorpalettes.ContinuousPalette = ContinuousPalette
    colorpalettes.Palette = Palette
    sys.modules[module_name] = colorpalettes


@st.cache_resource(show_spinner=False)
def load_model():
    install_orange_palette_shim()
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def finite_column(values: np.ndarray) -> np.ndarray:
    column = np.asarray(values, dtype=float)
    return column[np.isfinite(column)]


def timestamp_to_date(value: float) -> date:
    return datetime.fromtimestamp(float(value), tz=timezone.utc).date()


def money(value: float) -> str:
    return f"Rp {value:,.0f}".replace(",", ".")


def slider_bounds(values: dict[str, float], decimals: int) -> tuple[float, float, float]:
    factor = 10**decimals
    minimum = math.floor(values["min"] * factor) / factor
    maximum = math.ceil(values["max"] * factor) / factor
    default = round(values["median"], decimals)
    return minimum, maximum, default


def model_stats(model) -> dict[str, dict[str, float]]:
    instances = getattr(model, "instances", None)
    if instances is None:
        raise RuntimeError("Model pickle tidak menyimpan data latih untuk membuat slider.")

    stats: dict[str, dict[str, float]] = {}
    for index, variable in enumerate(model.domain.attributes):
        values = finite_column(instances.X[:, index])
        stats[variable.name] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        }
    return stats


def predict(model, feature_values: list[float]) -> float:
    from Orange.data import Table

    table = Table.from_list(model.domain, [feature_values])
    prediction = model(table)
    return float(np.asarray(prediction).reshape(-1)[0])


def main() -> None:
    st.set_page_config(
        page_title="Prediksi Harga Rumah",
        layout="centered",
    )

    st.title("Prediksi Harga Rumah")

    try:
        model = load_model()
        stats = model_stats(model)
    except Exception as exc:
        st.error(f"Model gagal dimuat: {exc}")
        st.stop()

    attributes = {variable.name: variable for variable in model.domain.attributes}
    house_age_min, house_age_max, house_age_default = slider_bounds(
        stats["X2 house age"], 1
    )
    mrt_min, mrt_max, mrt_default = slider_bounds(
        stats["X3 distance to the nearest MRT station"], 1
    )

    with st.form("prediction_form"):
        transaction_date = st.slider(
            "Tanggal transaksi",
            min_value=timestamp_to_date(stats["X1 transaction date"]["min"]),
            max_value=timestamp_to_date(stats["X1 transaction date"]["max"]),
            value=timestamp_to_date(stats["X1 transaction date"]["median"]),
            format="YYYY-MM-DD",
        )

        house_age = st.slider(
            "Umur rumah (tahun)",
            min_value=house_age_min,
            max_value=house_age_max,
            value=house_age_default,
            step=0.1,
        )

        mrt_distance = st.slider(
            "Jarak ke MRT terdekat (meter)",
            min_value=mrt_min,
            max_value=mrt_max,
            value=mrt_default,
            step=0.1,
        )

        convenience_stores = st.slider(
            "Jumlah convenience store",
            min_value=int(stats["X4 number of convenience stores"]["min"]),
            max_value=int(stats["X4 number of convenience stores"]["max"]),
            value=int(stats["X4 number of convenience stores"]["median"]),
            step=1,
        )

        latitude = st.slider(
            "Latitude",
            min_value=stats["X5 latitude"]["min"],
            max_value=stats["X5 latitude"]["max"],
            value=stats["X5 latitude"]["median"],
            step=0.00001,
            format="%.5f",
        )

        longitude = st.slider(
            "Longitude",
            min_value=stats["X6 longitude"]["min"],
            max_value=stats["X6 longitude"]["max"],
            value=stats["X6 longitude"]["median"],
            step=0.00001,
            format="%.5f",
        )

        submitted = st.form_submit_button("Prediksi", type="primary", use_container_width=True)

    if submitted:
        date_variable = attributes["X1 transaction date"]
        transaction_timestamp = float(date_variable.to_val(transaction_date.isoformat()))

        features = [
            transaction_timestamp,
            float(house_age),
            float(mrt_distance),
            float(convenience_stores),
            float(latitude),
            float(longitude),
        ]

        try:
            result = predict(model, features)
        except Exception as exc:
            st.error(f"Prediksi gagal: {exc}")
        else:
            st.metric("Estimasi harga rumah", money(result))
            st.caption(f"Target model: {getattr(model.domain.class_var, 'name', TARGET_LABEL)}")

if __name__ == "__main__":
    main()
