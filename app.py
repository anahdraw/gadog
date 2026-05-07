"""
Aplikasi Prediksi Harga Properti Taiwan
Berbasis model Orange Data Mining (Decision Tree Regressor)
Dijalankan di Streamlit Cloud - TANPA dependensi orange3

Strategi:
  Model .pickle dari Orange menyimpan objek Orange.tree.TreeModel.
  Daripada meng-install orange3 (yang butuh libglib2, PyQt, dll di server headless),
  kita reconstruct class stub yang cukup untuk membuat pickle.load() berhasil,
  lalu menjalankan inferensi Decision Tree secara native dengan NumPy.
"""

import sys
import types
import struct
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# ORANGE STUB SYSTEM
# Membuat module dummy untuk semua namespace Orange yang
# dibutuhkan pickle.load() agar tidak mencari orange3 asli.
# ─────────────────────────────────────────────────────────────

def _make_module(name):
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent_name = ".".join(parts[:i])
        parent = sys.modules.setdefault(parent_name, types.ModuleType(parent_name))
        setattr(parent, parts[i], mod)
    return mod


class _OrangeVariable:
    def __init__(self, *args, **kwargs):
        self.name   = args[0] if args else kwargs.get("name", "")
        self.values = kwargs.get("values", [])
    @classmethod
    def make(cls, *args, **kwargs):
        return cls(*args, **kwargs)
    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r})"

class _ContinuousVariable(_OrangeVariable):
    var_type = 2
    _max_round_diff = 1e-9
    _number_of_decimals = 1
    adjust_decimals = 0
    _format_str = "%.1f"

class _DiscreteVariable(_OrangeVariable):
    var_type = 1

class _TimeVariable(_OrangeVariable):
    var_type = 4
    _max_round_diff = 1e-9
    _number_of_decimals = 0
    adjust_decimals = 0
    _format_str = "%.0f"
    have_date = 1
    have_time = 0
    _timezone = None

class _StringVariable(_OrangeVariable):
    var_type = 3


def _make_variable(*args, **kwargs):
    cls  = args[0] if args else _ContinuousVariable
    name = args[1] if len(args) > 1 else ""
    return cls(name)


class _Domain:
    def __init__(self, *args, **kwargs):
        self.attributes = kwargs.get("attributes", ())
        self.class_vars = kwargs.get("class_vars", ())
        self.class_var  = self.class_vars[0] if self.class_vars else None
        self.metas      = kwargs.get("metas", ())
    def __getitem__(self, key):
        for v in list(self.attributes) + list(self.class_vars):
            if getattr(v, "name", None) == key:
                return v
        raise KeyError(key)

class _Table:
    def __init__(self):
        self.domain = None
        self.X = np.array([])
        self.Y = np.array([])
        self.metas = np.array([])

class _ContinuousPalette:
    def __init__(self, *a, **kw):
        self.name          = kw.get("name", "")
        self.friendly_name = kw.get("friendly_name", "")
        self.category      = kw.get("category", "")
        self.palette       = None


# ─────────────────────────────────────────────────────────────
# TREE MODEL STUB + NATIVE INFERENCE
# Orange TreeModel menyimpan pohon dalam dua array:
#   _thresholds : float64 array, nilai threshold tiap split
#   _code       : uint8  array, bytecode traversal pohon
#
# Format bytecode (Orange >= 3.28):
#   opcode=0  → LEAF  : 8 byte little-endian double (nilai prediksi)
#   opcode>0  → SPLIT : feature=(opcode-1),
#                        lalu threshold_index (varint),
#                        lalu right_branch_offset (varint)
# ─────────────────────────────────────────────────────────────

class _TreeModelStub:
    """Stub Orange.tree.TreeModel — bisa di-unpickle & menjalankan inferensi."""

    def __init__(self):
        self.domain               = None
        self._thresholds          = None
        self._code                = None
        self.supports_multiclass  = False

    def __reduce__(self):
        return (self.__class__, ())

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    # ── static helper ──
    @staticmethod
    def _read_varint(code, pos):
        result, shift = 0, 0
        while True:
            byte = int(code[pos]); pos += 1
            result |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                break
            shift += 7
        return result, pos

    def _traverse(self, row, code, thresholds):
        pos = 0
        for _ in range(50_000):
            if pos >= len(code):
                break
            opcode = int(code[pos]); pos += 1

            if opcode == 0:                               # LEAF
                return struct.unpack_from("<d", bytes(code[pos:pos+8]))[0]

            feature_idx          = opcode - 1            # SPLIT
            thr_idx, pos         = self._read_varint(code, pos)
            right_offset, pos    = self._read_varint(code, pos)
            threshold            = thresholds[thr_idx]
            feat_val             = float(row[feature_idx]) if feature_idx < len(row) else 0.0

            if feat_val > threshold:
                pos += right_offset
        return float("nan")

    def predict(self, X):
        X          = np.asarray(X, dtype=np.float64)
        code       = np.asarray(self._code,       dtype=np.uint8)
        thresholds = np.asarray(self._thresholds, dtype=np.float64)
        out        = np.empty(len(X), dtype=np.float64)
        for i, row in enumerate(X):
            out[i] = self._traverse(row, code, thresholds)
        return out

    def __call__(self, data):
        X = data.X if hasattr(data, "X") else np.array(data)
        return self.predict(X)


def _install_orange_stubs():
    var_mod = _make_module("Orange.data.variable")
    var_mod.ContinuousVariable = _ContinuousVariable
    var_mod.DiscreteVariable   = _DiscreteVariable
    var_mod.TimeVariable       = _TimeVariable
    var_mod.StringVariable     = _StringVariable
    var_mod.make_variable      = _make_variable

    dom_mod = _make_module("Orange.data.domain")
    dom_mod.Domain = _Domain

    data_mod = _make_module("Orange.data")
    data_mod.ContinuousVariable = _ContinuousVariable
    data_mod.DiscreteVariable   = _DiscreteVariable
    data_mod.TimeVariable       = _TimeVariable
    data_mod.StringVariable     = _StringVariable
    data_mod.Domain             = _Domain
    data_mod.Table              = _Table
    data_mod.make_variable      = _make_variable
    data_mod.variable           = var_mod
    data_mod.domain             = dom_mod

    tree_mod = _make_module("Orange.tree")
    tree_mod.TreeModel = _TreeModelStub

    cp_mod = _make_module("Orange.widgets.utils.colorpalettes")
    cp_mod.ContinuousPalette = _ContinuousPalette

    _make_module("Orange.base")
    _make_module("Orange")
    sys.modules["Orange"].data = data_mod
    sys.modules["Orange"].tree = tree_mod


# Install stub SEBELUM pickle.load dipanggil
_install_orange_stubs()


# ─────────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Harga Properti Taiwan",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "model_orange.pickle"

FEATURE_CONFIG = {
    "X1 transaction date": {
        "label": "X1 – Tanggal Transaksi",
        "type": "numeric", "input": "slider",
        "min": 2012.0, "max": 2014.0, "default": 2013.0,
        "step": 0.083, "format": "%.3f",
        "help": "Tahun transaksi dalam format desimal (contoh: 2013.500 = pertengahan 2013)",
    },
    "X2 house age": {
        "label": "X2 – Umur Bangunan (tahun)",
        "type": "numeric", "input": "slider",
        "min": 0.0, "max": 45.0, "default": 10.0,
        "step": 0.5, "format": "%.1f",
        "help": "Usia bangunan dalam tahun",
    },
    "X3 distance to the nearest MRT station": {
        "label": "X3 – Jarak ke Stasiun MRT (meter)",
        "type": "numeric", "input": "number",
        "min": 0.0, "max": 7000.0, "default": 500.0,
        "step": 10.0, "format": "%.2f",
        "help": "Jarak dari properti ke stasiun MRT terdekat (meter)",
    },
    "X4 number of convenience stores": {
        "label": "X4 – Jumlah Minimarket Terdekat",
        "type": "numeric", "input": "slider",
        "min": 0, "max": 10, "default": 5,
        "step": 1, "format": "%d",
        "help": "Jumlah minimarket dalam radius jalan kaki",
    },
    "X5 latitude": {
        "label": "X5 – Lintang (Latitude)",
        "type": "numeric", "input": "number",
        "min": 24.90, "max": 25.10, "default": 24.97,
        "step": 0.001, "format": "%.5f",
        "help": "Koordinat lintang lokasi properti (wilayah Taipei, Taiwan)",
    },
    "X6 longitude": {
        "label": "X6 – Bujur (Longitude)",
        "type": "numeric", "input": "number",
        "min": 121.45, "max": 121.60, "default": 121.53,
        "step": 0.001, "format": "%.5f",
        "help": "Koordinat bujur lokasi properti (wilayah Taipei, Taiwan)",
    },
}

FEATURE_ORDER = list(FEATURE_CONFIG.keys())


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Memuat model prediksi...")
def load_model():
    if not MODEL_PATH.exists():
        return None, (
            f"❌ File model tidak ditemukan: `{MODEL_PATH}`\n\n"
            "Pastikan file **model_orange.pickle** sudah ada di root repository "
            "GitHub yang sama dengan **app.py**."
        )
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model, None
    except Exception as e:
        return None, f"❌ Gagal memuat model: `{e}`"


# ─────────────────────────────────────────────────────────────
# PREDIKSI
# ─────────────────────────────────────────────────────────────
def predict_with_model(model, input_df):
    try:
        X      = input_df[FEATURE_ORDER].values.astype(np.float64)
        result = model.predict(X)
        pred   = float(result[0])
        if np.isnan(pred):
            return None, "⚠️ Model mengembalikan NaN. Periksa nilai input."
        return pred, None
    except Exception as e:
        return None, f"❌ Prediksi gagal: `{e}`"


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
def create_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/real-estate.png", width=72)
        st.title("Panduan Penggunaan")
        st.markdown("""
            **Langkah-langkah:**
            1. Atur nilai setiap fitur properti menggunakan kontrol di panel utama.
            2. Klik tombol **🏠 Prediksi Harga** untuk menjalankan model.
            3. Hasil prediksi akan ditampilkan di bawah form.

            ---
            **Tentang Model**

            Model ini adalah **Decision Tree Regressor** yang dilatih menggunakan
            **Orange Data Mining** pada dataset *Real Estate Valuation* (Taiwan).

            Inferensi dijalankan secara **native** tanpa instalasi orange3,
            menggunakan bytecode pohon yang tersimpan di dalam file pickle.

            ---
            **Target Prediksi**

            `Y – House Price of Unit Area`
            *(10.000 NTD per ping, 1 ping ≈ 3.3 m²)*
        """)
        st.divider()
        st.caption("📦 Model: Orange Decision Tree")
        st.caption("📊 Dataset: Taiwan Real Estate Valuation")
        st.caption("🚀 Deployed via Streamlit Cloud")


def _render_input(key, cfg, input_data):
    label     = cfg.get("label", key)
    help_text = cfg.get("help", "")
    if cfg["type"] == "categorical":
        val = st.selectbox(label, options=cfg["options"], help=help_text)
    elif cfg["input"] == "slider":
        val = st.slider(label,
            min_value=cfg["min"], max_value=cfg["max"],
            value=cfg["default"], step=cfg.get("step", 1), help=help_text)
    else:
        val = st.number_input(label,
            min_value=cfg["min"], max_value=cfg["max"],
            value=cfg["default"], step=cfg.get("step", 1.0),
            format=cfg.get("format", "%f"), help=help_text)
    input_data[key] = val


def create_input_form():
    input_data = {}
    with st.form("prediction_form"):
        st.subheader("📋 Input Data Properti")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            for key in FEATURE_ORDER[:3]:
                _render_input(key, FEATURE_CONFIG[key], input_data)
        with col2:
            for key in FEATURE_ORDER[3:]:
                _render_input(key, FEATURE_CONFIG[key], input_data)
        st.divider()
        submitted = st.form_submit_button(
            "🏠 Prediksi Harga", use_container_width=True, type="primary")
    return input_data, submitted


def display_results(input_data, pred_value):
    st.divider()
    st.subheader("📊 Ringkasan Input")
    display_df = pd.DataFrame(
        [{"Fitur": FEATURE_CONFIG[k]["label"], "Nilai": v}
         for k, v in input_data.items()])
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("🎯 Hasil Prediksi")
    col_res, col_info = st.columns([2, 3], gap="large")
    with col_res:
        st.success(
            f"**Harga per Unit Area yang Diprediksi:**\n\n"
            f"# {pred_value:,.2f}\n\n*(10.000 NTD / ping)*")
    with col_info:
        if   pred_value < 20: kat = "🟢 Murah – di bawah rata-rata pasar"
        elif pred_value < 40: kat = "🟡 Menengah – sekitar rata-rata pasar"
        elif pred_value < 60: kat = "🟠 Mahal – di atas rata-rata pasar"
        else:                 kat = "🔴 Sangat Mahal – jauh di atas rata-rata pasar"
        st.info(
            f"**Kategori Harga:** {kat}\n\n"
            "Nilai prediksi menggunakan Decision Tree Regressor "
            "yang dilatih pada dataset *Real Estate Valuation* Taiwan.\n\n"
            "Satuan harga: **10.000 NTD per ping** (1 ping ≈ 3.3 m²).")


def main():
    create_sidebar()
    st.title("🏠 Prediksi Harga Properti Taiwan")
    st.markdown(
        "Aplikasi ini menggunakan model **Decision Tree** hasil training dari "
        "**Orange Data Mining** yang dijalankan melalui **Streamlit Cloud**. "
        "Dataset yang digunakan adalah *Real Estate Valuation Data Set* "
        "dari wilayah Taipei, Taiwan.")
    st.divider()

    model, model_err = load_model()
    if model_err:
        st.error(model_err)
        st.stop()

    st.success("✅ Model berhasil dimuat dari `model_orange.pickle`")
    try:
        attrs = model.domain.attributes
        names = [getattr(a, "name", str(a)) for a in attrs]
        st.caption(f"🔍 Fitur model: {', '.join(names)}")
    except Exception:
        pass
    st.divider()

    input_data, submitted = create_input_form()
    if submitted:
        input_df = pd.DataFrame([input_data], columns=FEATURE_ORDER)
        with st.spinner("⚙️ Menjalankan prediksi..."):
            pred_value, pred_err = predict_with_model(model, input_df)
        if pred_err:
            st.error(pred_err)
            with st.expander("📌 Informasi Debug"):
                st.dataframe(input_df, use_container_width=True)
        else:
            display_results(input_data, pred_value)


if __name__ == "__main__":
    main()
