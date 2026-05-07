"""
Aplikasi Prediksi Harga Properti Taiwan
Berbasis model Orange Data Mining (Decision Tree Regressor)
Dijalankan di Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Harga Properti Taiwan",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# PATH MODEL — gunakan path relatif agar kompatibel dengan Streamlit Cloud
# Pastikan file model_orange.pickle ada di root repository GitHub yang sama
# ─────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "model_orange.pickle"

# ─────────────────────────────────────────────
# KONFIGURASI FITUR
# Nama fitur HARUS SAMA PERSIS dengan nama variabel saat training di Orange
# ─────────────────────────────────────────────
FEATURE_CONFIG = {
    "X1 transaction date": {
        "label": "X1 – Tanggal Transaksi",
        "type": "numeric",
        "input": "slider",
        "min": 2012.0,
        "max": 2014.0,
        "default": 2013.0,
        "step": 0.083,          # ≈ 1 bulan dalam satuan desimal tahun
        "format": "%.3f",
        "help": "Tahun transaksi dalam format desimal (contoh: 2013.500 = pertengahan 2013)",
    },
    "X2 house age": {
        "label": "X2 – Umur Bangunan (tahun)",
        "type": "numeric",
        "input": "slider",
        "min": 0.0,
        "max": 45.0,
        "default": 10.0,
        "step": 0.5,
        "format": "%.1f",
        "help": "Usia bangunan dalam tahun",
    },
    "X3 distance to the nearest MRT station": {
        "label": "X3 – Jarak ke Stasiun MRT (meter)",
        "type": "numeric",
        "input": "number",
        "min": 0.0,
        "max": 7000.0,
        "default": 500.0,
        "step": 10.0,
        "format": "%.2f",
        "help": "Jarak dari properti ke stasiun MRT terdekat (meter)",
    },
    "X4 number of convenience stores": {
        "label": "X4 – Jumlah Minimarket Terdekat",
        "type": "numeric",
        "input": "slider",
        "min": 0,
        "max": 10,
        "default": 5,
        "step": 1,
        "format": "%d",
        "help": "Jumlah minimarket/convenience store dalam radius jalan kaki",
    },
    "X5 latitude": {
        "label": "X5 – Lintang (Latitude)",
        "type": "numeric",
        "input": "number",
        "min": 24.90,
        "max": 25.10,
        "default": 24.97,
        "step": 0.001,
        "format": "%.5f",
        "help": "Koordinat lintang lokasi properti (wilayah Taipei, Taiwan)",
    },
    "X6 longitude": {
        "label": "X6 – Bujur (Longitude)",
        "type": "numeric",
        "input": "number",
        "min": 121.45,
        "max": 121.60,
        "default": 121.53,
        "step": 0.001,
        "format": "%.5f",
        "help": "Koordinat bujur lokasi properti (wilayah Taipei, Taiwan)",
    },
}

# ─────────────────────────────────────────────
# LOAD MODEL dengan st.cache_resource
# Agar model tidak di-reload setiap interaksi user
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Memuat model prediksi...")
def load_model():
    """
    Memuat model dari file pickle.
    Mengembalikan (model, error_message).
    """
    if not MODEL_PATH.exists():
        return None, (
            f"❌ File model tidak ditemukan: `{MODEL_PATH}`\n\n"
            "Pastikan file **model_orange.pickle** sudah ada di root repository GitHub "
            "yang sama dengan **app.py**."
        )
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model, None
    except Exception as e:
        return None, f"❌ Gagal memuat model: {e}"


# ─────────────────────────────────────────────
# FUNGSI PREDIKSI
# ─────────────────────────────────────────────
def predict_with_model(model, input_df: pd.DataFrame):
    """
    Menjalankan prediksi. Mencoba pendekatan scikit-learn terlebih dulu,
    lalu fallback ke format Orange jika gagal.
    """
    # --- Pendekatan 1: scikit-learn-compatible (.predict) ---
    try:
        result = model.predict(input_df)
        return float(result[0]), None
    except Exception as sklearn_err:
        pass  # lanjut ke fallback Orange

    # --- Pendekatan 2: Orange native ---
    return predict_with_orange_fallback(model, input_df)


def predict_with_orange_fallback(model, input_df: pd.DataFrame):
    """
    Fallback: konversi DataFrame ke Orange.data.Table lalu jalankan model.
    """
    try:
        import Orange.data as od

        # Buat daftar domain variabel
        attrs = []
        for col in input_df.columns:
            cfg = FEATURE_CONFIG[col]
            if cfg["type"] == "categorical":
                attrs.append(od.DiscreteVariable(col, values=cfg["options"]))
            else:
                # Kolom X1 adalah TimeVariable di Orange, tapi ContinuousVariable
                # sudah cukup untuk prediksi numerik
                attrs.append(od.ContinuousVariable(col))

        domain = od.Domain(attrs)

        # Konversi nilai ke numpy array
        row_values = []
        for col in input_df.columns:
            cfg = FEATURE_CONFIG[col]
            val = input_df[col].iloc[0]
            if cfg["type"] == "categorical":
                var = domain[col]
                row_values.append(float(var.to_val(str(val))))
            else:
                row_values.append(float(val))

        x = np.array([row_values])
        orange_table = od.Table.from_numpy(domain, x)

        predictions = model(orange_table)

        # Ambil nilai prediksi (bisa array atau scalar)
        if hasattr(predictions, "__len__"):
            pred_value = float(predictions[0])
        else:
            pred_value = float(predictions)

        return pred_value, None

    except ImportError:
        return None, (
            "❌ Library **orange3** tidak tersedia di environment ini.\n\n"
            "Pastikan `orange3` ada di **requirements.txt** repository Anda."
        )
    except Exception as e:
        return None, f"❌ Prediksi dengan Orange gagal: `{e}`"


# ─────────────────────────────────────────────
# KOMPONEN UI
# ─────────────────────────────────────────────
def create_sidebar():
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/real-estate.png",
            width=72,
        )
        st.title("Panduan Penggunaan")
        st.markdown(
            """
            **Langkah-langkah:**
            1. Atur nilai setiap fitur properti menggunakan kontrol di panel utama.
            2. Klik tombol **🏠 Prediksi Harga** untuk menjalankan model.
            3. Hasil prediksi akan ditampilkan di bawah form.

            ---
            **Tentang Model**

            Model ini adalah **Decision Tree Regressor** yang dilatih menggunakan
            **Orange Data Mining** pada dataset *Real Estate Valuation* (Taiwan).

            Model disimpan sebagai file `model_orange.pickle` dan dimuat langsung
            dari **GitHub repository** yang sama dengan `app.py`.

            ---
            **Target Prediksi**

            `Y – House Price of Unit Area`
            *(10.000 NTD per ping)*

            Semakin tinggi nilainya, semakin mahal harga per satuan luas properti.
            """
        )
        st.divider()
        st.caption("📦 Model: Orange Decision Tree")
        st.caption("📊 Dataset: Taiwan Real Estate Valuation")
        st.caption("🚀 Deployed via Streamlit Cloud")


def create_input_form():
    """
    Membuat form input berdasarkan FEATURE_CONFIG.
    Mengembalikan (input_dict, submitted).
    """
    input_data = {}

    with st.form("prediction_form"):
        st.subheader("📋 Input Data Properti")

        col1, col2 = st.columns(2, gap="large")

        feature_keys = list(FEATURE_CONFIG.keys())
        left_keys  = feature_keys[:3]
        right_keys = feature_keys[3:]

        # Kolom kiri
        with col1:
            for key in left_keys:
                cfg = FEATURE_CONFIG[key]
                _render_input(key, cfg, input_data)

        # Kolom kanan
        with col2:
            for key in right_keys:
                cfg = FEATURE_CONFIG[key]
                _render_input(key, cfg, input_data)

        st.divider()
        submitted = st.form_submit_button(
            "🏠 Prediksi Harga",
            use_container_width=True,
            type="primary",
        )

    return input_data, submitted


def _render_input(key: str, cfg: dict, input_data: dict):
    """Helper untuk render satu widget input sesuai konfigurasi."""
    label = cfg.get("label", key)
    help_text = cfg.get("help", "")

    if cfg["type"] == "categorical":
        val = st.selectbox(label, options=cfg["options"], help=help_text)

    elif cfg["input"] == "slider":
        val = st.slider(
            label,
            min_value=cfg["min"],
            max_value=cfg["max"],
            value=cfg["default"],
            step=cfg.get("step", 1),
            help=help_text,
        )

    elif cfg["input"] == "number":
        val = st.number_input(
            label,
            min_value=cfg["min"],
            max_value=cfg["max"],
            value=cfg["default"],
            step=cfg.get("step", 1.0),
            format=cfg.get("format", "%f"),
            help=help_text,
        )
    else:
        val = cfg["default"]

    input_data[key] = val


def display_results(input_data: dict, pred_value: float):
    """Menampilkan tabel input + hasil prediksi."""
    st.divider()

    # Tabel ringkasan input
    st.subheader("📊 Ringkasan Input")
    display_df = pd.DataFrame(
        [
            {
                "Fitur": FEATURE_CONFIG[k]["label"],
                "Nilai": v,
            }
            for k, v in input_data.items()
        ]
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Hasil prediksi
    st.subheader("🎯 Hasil Prediksi")
    col_res, col_info = st.columns([2, 3], gap="large")

    with col_res:
        st.success(
            f"**Harga per Unit Area yang Diprediksi:**\n\n"
            f"# {pred_value:,.2f}\n\n"
            f"*(10.000 NTD / ping)*"
        )

    with col_info:
        # Interpretasi sederhana
        if pred_value < 20:
            kategori = "🟢 Murah – di bawah rata-rata pasar"
        elif pred_value < 40:
            kategori = "🟡 Menengah – sekitar rata-rata pasar"
        elif pred_value < 60:
            kategori = "🟠 Mahal – di atas rata-rata pasar"
        else:
            kategori = "🔴 Sangat Mahal – jauh di atas rata-rata pasar"

        st.info(
            f"**Kategori Harga:** {kategori}\n\n"
            "Nilai prediksi menggunakan Decision Tree Regressor "
            "yang dilatih pada dataset *Real Estate Valuation* Taiwan.\n\n"
            "Satuan harga: **10.000 NTD per ping** (1 ping ≈ 3.3 m²)."
        )


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    # Sidebar
    create_sidebar()

    # Header
    st.title("🏠 Prediksi Harga Properti Taiwan")
    st.markdown(
        "Aplikasi ini menggunakan model **Decision Tree** hasil training dari "
        "**Orange Data Mining** yang dijalankan melalui **Streamlit Cloud**. "
        "Dataset yang digunakan adalah *Real Estate Valuation Data Set* dari wilayah Taipei, Taiwan."
    )
    st.divider()

    # Load model
    model, model_err = load_model()

    if model_err:
        st.error(model_err)
        st.stop()

    st.success("✅ Model berhasil dimuat dari `model_orange.pickle`")

    # Tampilkan info model Orange jika bisa dibaca
    try:
        domain = model.domain
        feature_names = [v.name for v in domain.attributes]
        st.caption(f"🔍 Fitur model: {', '.join(feature_names)}")
    except Exception:
        pass  # tidak kritis jika domain tidak bisa dibaca

    st.divider()

    # Form input & prediksi
    input_data, submitted = create_input_form()

    if submitted:
        # Susun DataFrame dengan urutan kolom sesuai FEATURE_CONFIG
        input_df = pd.DataFrame(
            [input_data], columns=list(FEATURE_CONFIG.keys())
        )

        with st.spinner("⚙️ Menjalankan prediksi..."):
            pred_value, pred_err = predict_with_model(model, input_df)

        if pred_err:
            st.error(pred_err)
            with st.expander("📌 Informasi Debug"):
                st.dataframe(input_df, use_container_width=True)
                st.markdown(
                    "**Tips:** Pastikan nama kolom di `FEATURE_CONFIG` "
                    "**sama persis** dengan nama variabel saat training di Orange."
                )
        else:
            display_results(input_data, pred_value)


if __name__ == "__main__":
    main()
