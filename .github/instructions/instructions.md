---
applyTo: '**'
---

> Kamu adalah AI engineer assistant yang bertugas membangun project riset berjudul **"Model Prediksi Gas Fee Ethereum Berbasis Hybrid LSTM–XGBoost untuk Optimasi Biaya Transaksi Pasca-EIP-1559"**.
>
> Tujuan proyek ini adalah membuat **pipeline headless (tanpa GUI)** yang berjalan di **testnet Sepolia**, untuk mengambil data blok Ethereum, membangun fitur, melatih model machine learning (LSTM + XGBoost hybrid), melakukan prediksi gas fee berikutnya, serta mengevaluasi performanya berdasarkan efisiensi biaya dan tingkat keberhasilan transaksi.
>
> Fokus proyek:
>
> * Eksperimen akademik (skripsi) dan bukti implementasi industri (efisiensi biaya transaksi).
> * Gunakan bahasa Python 3.10+ dan library umum seperti: `requests`, `pandas`, `numpy`, `xgboost`, `torch`, `scikit-learn`, `matplotlib`, `yaml`.
> * Tidak ada GUI; hanya CLI berbasis modul.
> * Jalankan seluruh pipeline secara modular dan terdokumentasi dengan baik.
> * **Jaga workspace tetap CLEAN, COMPACT, dan ORGANIZED**.

---

## 🎯 **Tugas Utama**

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

## 📁 **Struktur Folder dan File (Clean & Compact)**

```
gas-ml/
  cfg/exp.yaml                    # Configuration
  cli.py                          # ⭐ Main CLI entry point
  run.ps1                         # ⭐ Unified PowerShell runner
  
  src/
    rpc.py                        # RPC client
    fetch.py                      # Data fetcher
    features.py                   # Feature engineering
    feature_selector.py           # Unified feature selection (analyze/importance/filter)
    train.py                      # Training pipeline (supports --selected-features)
    infer.py                      # Inference engine
    policy.py                     # Fee recommendation policy
    evaluate.py                   # Evaluation metrics
    stack.py                      # Hybrid LSTM→XGBoost
    
  data/                           # Datasets (CSV/Parquet)
  models/                         # Trained models
  reports/                        # Evaluation results
  notebooks/                      # Jupyter notebooks for analysis
  tests/                          # Unit tests
  docs/                           # Essential documentation only
  
  .env.example
  .gitignore
  requirements.txt
  README.md                       # Single comprehensive documentation
  QUICK_REFERENCE.md              # Quick command reference
```

---

## ⚙️ **Pipeline Eksekusi (Unified CLI)**

### **Production CLI** (Recommended):
```bash
# Real-time prediction
python cli.py predict --rpc <RPC_URL>
python cli.py predict --rpc <RPC_URL> --continuous

# Model info
python cli.py info

# Backtesting
python cli.py backtest --data data/blocks_5k.csv
```

### **PowerShell Unified Runner**:
```powershell
.\run.ps1 fetch -NBlocks 5000        # Fetch data
.\run.ps1 features                   # Generate features
.\run.ps1 select-features            # Feature selection
.\run.ps1 train                      # Train model
.\run.ps1 train -SelectedFeatures "data\selected_features.txt"  # Train with selected features
.\run.ps1 predict                    # Real-time prediction
.\run.ps1 backtest                   # Run backtest
.\run.ps1 test                       # Run unit tests
.\run.ps1 info                       # Model information
```

### **Python Module Commands**:
```bash
# Data pipeline
python -m src.fetch --network sepolia --n-blocks 8000
python -m src.features --in data/blocks.csv --out data/features.parquet

# Feature selection (Unified module with subcommands)
python -m src.feature_selector analyze --in data/features.parquet
python -m src.feature_selector importance --model models/xgb.bin --in data/features.parquet
python -m src.feature_selector filter --in data/features.parquet --features data/selected_features.txt --out data/filtered.parquet

# Training (Enhanced with feature selection support)
python -m src.train --cfg cfg/exp.yaml --in data/features.parquet
python -m src.train --cfg cfg/exp.yaml --in data/features.parquet --selected-features data/selected_features.txt

# Inference
python -m src.infer --model models/
python -m src.infer --model models/ --continuous --interval 12
```

---

### 📊 **Output yang Diharapkan**

* Dataset blok & fitur (CSV/Parquet).
* Model tersimpan (`models/xgb.bin`, `models/lstm.pt`, `models/hybrid.pkl`).
* File evaluasi (`reports/metrics.json`, `reports/backtest.csv`).
* CLI real-time rekomendasi gas fee.

---

---

## ⚠️ **CRITICAL RULES - MUST FOLLOW**

### **🚫 NEVER CREATE SUMMARY/DOCUMENTATION FILES**

**ABSOLUTELY FORBIDDEN:**
- ❌ NO `CLEANUP_SUMMARY.md`
- ❌ NO `QUICK_REFERENCE.md`
- ❌ NO `CHANGES.md`
- ❌ NO `SUMMARY.md`
- ❌ NO `MIGRATION_GUIDE.md`
- ❌ NO any other `.md` files documenting changes

**ONLY 2 MARKDOWN FILES ALLOWED:**
- ✅ `README.md` (main project documentation in gas-ml/ - MUST EXIST)
- ✅ `instructions.md` (this file in .github/instructions/)

**Technical docs in `docs/` folder are ALLOWED but NOT .md files:**
- Use existing docs/CLI_USAGE.md, docs/GPU_TRAINING.md (keep if exist, don't create new)

**WHY?**
Documentation clutter defeats the purpose of cleanup. Changes should be self-evident from:
1. Git commit messages
2. Code comments
3. Updated README.md

**IF YOU CREATE A SUMMARY .MD FILE, YOU FAILED THE TASK.**

---

## 🧹 **Prinsip Clean Workspace**

### **✅ DO (Lakukan)**:
1. **Unify similar functionality** - Jika ada 2-3 script yang mirip, gabungkan jadi satu dengan flags/subcommands
2. **Single entry point** - Gunakan `cli.py` untuk production, `run.ps1` untuk development
3. **Consolidate documentation** - Satu README comprehensive, bukan 3-4 docs terpisah
4. **Remove redundancy** - Hapus file duplikat, archive lama, atau unused code
5. **Use subcommands** - `feature_selector.py analyze/importance/filter` lebih baik dari 3 file terpisah
6. **Keep structure flat** - Hindari nested folders yang tidak perlu
7. **Proper .gitignore** - Ignore `__pycache__`, `venv/`, temporary files

### **❌ DON'T (Jangan)**:
1. **Jangan buat file baru** untuk setiap small variant - gunakan flags/parameters
2. **Jangan simpan archive** di workspace - pindahkan ke backup eksternal atau hapus
3. **Jangan duplikasi logic** - reuse functions, jangan copy-paste
4. **Jangan buat banyak docs** - one comprehensive is better than many scattered
5. **Jangan biarkan old scripts** tertinggal setelah refactor
6. **Jangan commit temporary files** - gunakan .gitignore
7. **Jangan buat nested subdirectories** tanpa alasan jelas
8. **🚫 JANGAN PERNAH BUAT SUMMARY/CLEANUP DOCUMENTATION FILES** - Use git commits instead

### **🔄 Cleanup Checklist**:
```
[ ] Identify duplicate/similar files
[ ] Merge into unified modules with subcommands
[ ] Remove old/unused files
[ ] Consolidate documentation into README.md
[ ] Clean data/ directory (remove archives)
[ ] Update .gitignore
[ ] Remove __pycache__ directories
[ ] Unify CLI interfaces
[ ] Simplify script runners
[ ] 🚫 DO NOT create any summary/cleanup .md files
[ ] Commit changes with clear git commit message
```

### **Example Cleanup Actions**:
```powershell
# Remove archives
Remove-Item -Path "data/archive" -Recurse -Force

# Remove duplicate scripts
Remove-Item -Path "src/train_selected.py" -Force
Remove-Item -Path "src/feature_analysis.py", "src/feature_selection.py", "src/filter_features.py" -Force

# Remove old PowerShell scripts
Remove-Item -Path "scripts/run_*.ps1" -Force

# Remove duplicate docs
Remove-Item -Path "PROJECT_SUMMARY.md", "SETUP.md", "CHANGELOG.md" -Force

# Clean Python cache
Get-ChildItem -Path . -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force
```

---

## 🧾 **Catatan Tambahan**

* Jalankan semua script dalam environment Python tanpa GUI.
* Gunakan testnet **Sepolia** agar bebas biaya gas.
* Dokumentasikan setiap fungsi dengan docstring jelas (biar mudah dijelaskan di sidang).
* Gunakan logging sederhana untuk setiap tahap pipeline.
* Hindari library eksperimental; prioritaskan stabilitas.
* **Maintain clean workspace** - regular cleanup untuk menghindari clutter.
* **One source of truth** - satu file untuk satu purpose, hindari duplikasi.

---

## 🧩 **Tujuan Akhir Proyek**

> Membangun sistem cerdas yang mampu memprediksi tren gas fee Ethereum secara real-time untuk memberikan rekomendasi biaya transaksi optimal, menggantikan tebakan manual pengguna dan meningkatkan efisiensi ekonomi di jaringan blockchain.
>
> **Dengan workspace yang CLEAN, COMPACT, dan WELL-ORGANIZED untuk kemudahan maintenance dan presentasi.**

---

## 📝 **Version & Status**

* **Version**: 2.0 (Clean & Compact Edition)
* **Status**: ✅ Production Ready
* **Last Updated**: November 2025
* **Structure**: Unified & Optimized

