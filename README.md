# Prediksi Harga Rumah Streamlit

Aplikasi ini memuat `treehouse.pkcls`, model Orange TreeModel untuk memprediksi `Y house price of unit area` dari enam input:

- `X1 transaction date`
- `X2 house age`
- `X3 distance to the nearest MRT station`
- `X4 number of convenience stores`
- `X5 latitude`
- `X6 longitude`

## Jalankan lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Push folder ini ke repository GitHub.
2. Pastikan file berikut ikut ter-upload: `app.py`, `requirements.txt`, `runtime.txt`, `.streamlit/config.toml`, dan `treehouse.pkcls`.
3. Di Streamlit Community Cloud, pilih repository tersebut.
4. Set main file path ke `app.py`.
5. Deploy.

Catatan: pickle ini berasal dari Orange dan menyimpan metadata palette widget. `app.py` memasang shim kecil agar model bisa dimuat di Streamlit tanpa memasang dependency Qt.
