---
applyTo: '**'
---

> Kamu adalah AI engineer assistant yang bertugas membangun project riset berjudul **“Model Prediksi Gas Fee Ethereum Berbasis Hybrid LSTM–XGBoost untuk Optimasi Biaya Transaksi Pasca-EIP-1559”**.
>
> Tujuan proyek ini adalah membuat **pipeline headless (tanpa GUI)** yang berjalan di **testnet Sepolia**, untuk mengambil data blok Ethereum, membangun fitur, melatih model machine learning (LSTM + XGBoost hybrid), melakukan prediksi gas fee berikutnya, serta mengevaluasi performanya berdasarkan efisiensi biaya dan tingkat keberhasilan transaksi.
>
> Fokus proyek:
>
> * Eksperimen akademik (skripsi) dan bukti implementasi industri (efisiensi biaya transaksi).
> * Gunakan bahasa Python 3.10+ dan library umum seperti: `requests`, `pandas`, `numpy`, `xgboost`, `torch`, `scikit-learn`, `matplotlib`, `yaml`.
> * Tidak ada GUI; hanya CLI berbasis modul.
> * Jalankan seluruh pipeline secara modular dan terdokumentasi dengan baik.

---

### 🎯 **Tugas Utama**

1. **Ambil data blok dari testnet Sepolia (RPC / Etherscan API)**
   Kolom wajib: `number`, `timestamp`, `baseFeePerGas`, `gasUsed`, `gasLimit`, `txCount`.
2. **Bangun fitur**: `ΔbaseFee`, `utilization = gasUsed/gasLimit`, `EMA_baseFee`, `EMA_utilization`, `hour_of_day`, `day_of_week`.
3. **Buat label**: `baseFee_next` (blok berikutnya).
4. **Latih model hybrid**:

   * LSTM → ekstraksi pola temporal.
   * XGBoost → regresi akhir (stacking).
5. **Evaluasi model** menggunakan: `MAE`, `MAPE`, `under-estimation rate`, `Hit@ε`, dan `cost-saving`.
6. **Implementasi inferensi real-time**: ambil blok terbaru → prediksi gas fee berikutnya → rekomendasikan `priorityFee`, `buffer`, `maxFee`.
7. **Backtesting**: bandingkan hasil prediksi vs baseline EIP-1559.
8. **Output**: CLI rekomendasi seperti:

   ```
   BaseFee_pred: 22.7 Gwei
   PriorityFee_suggested: 1.3 Gwei
   MaxFee_recommendation: 24.5 Gwei
   Status: Optimal – Ready to Broadcast
   ```

---

### 📁 **Struktur Folder dan File yang Harus Dibuat**

```
gas-ml/
  cfg/exp.yaml
  data/
  models/
  src/
    rpc.py          # akses RPC Sepolia
    fetch.py        # ambil blok → CSV
    features.py     # fitur & label
    train.py        # latih model
    infer.py        # prediksi CLI
    policy.py       # rekomendasi tip/buffer
    backtest.py     # simulasi cost-saving
    evaluate.py     # metrik evaluasi
    stack.py        # wrapper LSTM→XGBoost
  scripts/
    run_fetch.sh
    run_train.sh
    run_infer.sh
  tests/
    test_metrics.py
  .env.example
  requirements.txt
  README.md
```

---

### ⚙️ **Pipeline Eksekusi (CLI)**

```bash
python -m src.fetch --network sepolia --n-blocks 8000
python -m src.features --in data/blocks.csv --out data/features.parquet
python -m src.train --cfg cfg/exp.yaml --in data/features.parquet
python -m src.infer --model models/hybrid.bin
python -m src.backtest --model models/hybrid.bin --data data/features.parquet
python -m src.evaluate --pred data/pred.csv --true data/features.parquet
```

---

### 📊 **Output yang Diharapkan**

* Dataset blok & fitur (CSV/Parquet).
* Model tersimpan (`models/xgb.bin`, `models/lstm.pt`, `models/hybrid.pkl`).
* File evaluasi (`reports/metrics.json`, `reports/backtest.csv`).
* CLI real-time rekomendasi gas fee.

---

### 🧾 **Catatan Tambahan**

* Jalankan semua script dalam environment Python tanpa GUI.
* Gunakan testnet **Sepolia** agar bebas biaya gas.
* Dokumentasikan setiap fungsi dengan docstring jelas (biar mudah dijelaskan di sidang).
* Gunakan logging sederhana untuk setiap tahap pipeline.
* Hindari library eksperimental; prioritaskan stabilitas.

---

🧩 **Tujuan akhir proyek:**

> Membangun sistem cerdas yang mampu memprediksi tren gas fee Ethereum secara real-time untuk memberikan rekomendasi biaya transaksi optimal, menggantikan tebakan manual pengguna dan meningkatkan efisiensi ekonomi di jaringan blockchain.

