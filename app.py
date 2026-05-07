import sys
import types
import importlib.abc
import importlib.machinery
from unittest.mock import MagicMock

# ─────────────────────────────────────────────────────────────
# Orange3 menggunakan Qt dan orangecanvas untuk modul widgets-nya.
# Kita hanya butuh bagian prediksi (Orange.tree, Orange.data),
# sehingga semua import dari orangecanvas.* dan AnyQt.*
# di-stub agar tidak memerlukan Qt / GUI dependencies.
#
# PENTING: orangecanvas.localization.si berisi fungsi plsi(),
# plsi_sz(), z_besedo() yang dipanggil saat definisi kelas
# di Orange.data.variable. Fungsi-fungsi ini HARUS mengembalikan
# string asli (bukan MagicMock), agar class attributes terbentuk
# dengan benar dan pickle bisa me-reconstruct objek model.
# ─────────────────────────────────────────────────────────────

_MOCK_PREFIXES = ("orangecanvas", "AnyQt", "Orange.widgets")

# ── Buat stub modul orangecanvas.localization.si ────────────
# Fungsi-fungsi ini dipakai untuk i18n/pluralisasi di Orange3.
# Kita cukup kembalikan string apa adanya tanpa terjemahan.


def _plsi(n, *forms):
    """Stub pluralisasi: kembalikan form pertama."""
    return forms[0] if forms else ""


def _plsi_sz(n, *forms):
    """Stub pluralisasi dengan angka."""
    return f"{n} {forms[0]}" if forms else str(n)


def _z_besedo(*args):
    """Stub z_besedo: kembalikan argumen pertama."""
    return args[0] if args else ""


# Bangun modul orangecanvas.localization.si secara manual
_oc = types.ModuleType("orangecanvas")
_oc.__path__ = []
_oc.__package__ = "orangecanvas"

_oc_loc = types.ModuleType("orangecanvas.localization")
_oc_loc.__path__ = []
_oc_loc.__package__ = "orangecanvas"

_oc_loc_si = types.ModuleType("orangecanvas.localization.si")
_oc_loc_si.__package__ = "orangecanvas.localization"
_oc_loc_si.plsi = _plsi
_oc_loc_si.plsi_sz = _plsi_sz
_oc_loc_si.z_besedo = _z_besedo

# Hubungkan parent ↔ child
_oc.localization = _oc_loc
_oc_loc.si = _oc_loc_si

sys.modules["orangecanvas"] = _oc
sys.modules["orangecanvas.localization"] = _oc_loc
sys.modules["orangecanvas.localization.si"] = _oc_loc_si

# ── Inject MagicMock untuk modul GUI lain ───────────────────
_OTHER_MOCKS = [
    "AnyQt", "AnyQt.QtCore", "AnyQt.QtGui", "AnyQt.QtWidgets",
    "Orange.widgets",
]

for _mod_name in _OTHER_MOCKS:
    if _mod_name not in sys.modules:
        _mock = MagicMock()
        _mock.__name__ = _mod_name
        _mock.__package__ = _mod_name.rpartition(".")[0]
        _mock.__path__ = []
        _mock.__spec__ = None
        sys.modules[_mod_name] = _mock


# ── Meta-path finder sebagai fallback ───────────────────────

class _AutoMockFinder(importlib.abc.MetaPathFinder):
    """Intercept import dari namespace tertentu dan kembalikan MagicMock."""

    def find_spec(self, fullname, path, target=None):
        if fullname in sys.modules:
            return None
        if any(fullname == p or fullname.startswith(p + ".") for p in _MOCK_PREFIXES):
            return importlib.machinery.ModuleSpec(fullname, _AutoMockLoader())
        return None


class _AutoMockLoader(importlib.abc.Loader):
    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        mod.__loader__ = self
        mod.__package__ = spec.name.rpartition(".")[0]
        mod.__spec__ = spec
        mod.__path__ = []
        sys.modules[spec.name] = mod
        return mod

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _AutoMockFinder())

import pickle
import datetime
import numpy as np
import streamlit as st
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Konfigurasi halaman
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Harga Rumah",
    page_icon="🏠",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────
# Load model Orange3 (TreeModel / Decision Tree Regressor)
# File pickle harus ada di folder yang sama dengan app.py
# ─────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "treehouse.pkcls"

@st.cache_resource(show_spinner="Memuat model prediksi…")
def load_model():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.title("🏠 Prediksi Harga Rumah")
st.markdown(
    "Isi detail properti di bawah ini, lalu tekan **Prediksi Harga** "
    "untuk mendapatkan estimasi harga rumah."
)
st.divider()

if model is None:
    st.error(
        "⚠️ File model **treehouse.pkcls** tidak ditemukan.\n\n"
        "Pastikan file tersebut berada di folder yang sama dengan `app.py`."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────
# Form Input
# ─────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    st.subheader("📋 Detail Properti")

    col1, col2 = st.columns(2)

    with col1:
        transaction_date = st.date_input(
            "📅 Tanggal Transaksi",
            value=datetime.date(2013, 6, 1),
            min_value=datetime.date(2012, 1, 1),
            max_value=datetime.date(2030, 12, 31),
            help="Tanggal transaksi jual-beli properti",
        )
        house_age = st.number_input(
            "🏗️ Usia Rumah (tahun)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.1,
            format="%.1f",
            help="Usia bangunan rumah dalam tahun",
        )
        distance_mrt = st.number_input(
            "🚇 Jarak ke Stasiun MRT (meter)",
            min_value=0.0,
            max_value=10_000.0,
            value=500.0,
            step=10.0,
            format="%.1f",
            help="Jarak dari properti ke stasiun MRT terdekat",
        )

    with col2:
        convenience_stores = st.number_input(
            "🏪 Jumlah Minimarket Terdekat",
            min_value=0,
            max_value=20,
            value=3,
            step=1,
            help="Jumlah minimarket / convenience store dalam radius dekat",
        )
        latitude = st.number_input(
            "🌐 Latitude",
            min_value=24.0,
            max_value=26.0,
            value=24.9675,
            step=0.0001,
            format="%.4f",
            help="Koordinat lintang lokasi properti",
        )
        longitude = st.number_input(
            "🌐 Longitude",
            min_value=121.0,
            max_value=122.0,
            value=121.5333,
            step=0.0001,
            format="%.4f",
            help="Koordinat bujur lokasi properti",
        )

    st.divider()
    submitted = st.form_submit_button(
        "🔍 Prediksi Harga",
        use_container_width=True,
        type="primary",
    )

# ─────────────────────────────────────────────────────────────
# Prediksi
# ─────────────────────────────────────────────────────────────
if submitted:
    try:
        from Orange.data import Table

        # Konversi tanggal ke Unix timestamp (yang dipakai TimeVariable Orange3)
        dt = datetime.datetime.combine(transaction_date, datetime.time(0, 0, 0))
        timestamp = dt.timestamp()

        # Susun array fitur sesuai urutan domain model:
        # X1 transaction date, X2 house age, X3 distance MRT,
        # X4 convenience stores, X5 latitude, X6 longitude
        X = np.array([[
            timestamp,
            float(house_age),
            float(distance_mrt),
            float(convenience_stores),
            float(latitude),
            float(longitude),
        ]])

        # Buat Orange Table menggunakan domain model (tanpa class / Y)
        domain_no_class = model.domain.copy()
        instance = Table.from_numpy(model.domain, X, None)

        # Jalankan prediksi
        prediction = model(instance)
        harga = float(prediction[0])

        # ── Tampilkan hasil ──────────────────────────────────
        st.success("✅ Prediksi berhasil!")
        st.metric(
            label="Estimasi Harga per Unit Area",
            value=f"Rp {harga:,.0f}",
        )

        with st.expander("📊 Detail Input yang Digunakan"):
            st.table(
                {
                    "Fitur": [
                        "Tanggal Transaksi",
                        "Usia Rumah",
                        "Jarak ke MRT",
                        "Jumlah Minimarket",
                        "Latitude",
                        "Longitude",
                    ],
                    "Nilai": [
                        str(transaction_date),
                        f"{house_age:.1f} tahun",
                        f"{distance_mrt:,.1f} meter",
                        str(int(convenience_stores)),
                        f"{latitude:.4f}",
                        f"{longitude:.4f}",
                    ],
                }
            )

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat prediksi:\n\n`{e}`")
        st.info(
            "Pastikan file **treehouse.pkcls** adalah model Orange3 yang valid "
            "dan sesuai dengan fitur yang diharapkan."
        )

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: Orange3 Decision Tree Regressor • "
    "Dataset: Prediksi Harga Rumah • "
    "App dibuat dengan Streamlit"
)
