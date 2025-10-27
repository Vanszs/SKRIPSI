# Changelog

All notable changes to the Gas Fee Prediction Model project will be documented in this file.

## [1.0.0] - 2025-01-27

### Added
- ✨ Initial implementation of hybrid LSTM→XGBoost model
- 📊 Complete data fetching pipeline from Sepolia testnet
- 🔧 Comprehensive feature engineering module
- 🤖 Training pipeline with configurable hyperparameters
- 🎯 Real-time inference engine with continuous mode
- 💰 Gas fee policy recommender with dynamic buffer
- 📈 Backtesting framework with cost-saving analysis
- ✅ Evaluation metrics (MAE, MAPE, RMSE, Hit@ε, cost-saving)
- 📜 PowerShell automation scripts for Windows
- 🧪 Unit tests for critical components
- 📚 Comprehensive documentation (README, SETUP, API docs)

### Features
- RPC client dengan retry mechanism dan fallback endpoints
- Feature engineering: temporal features, EMA, rolling statistics
- LSTM feature extractor for temporal patterns
- XGBoost meta-learner for final predictions
- Policy-based gas fee recommendations
- Confidence scoring for predictions
- Historical backtesting vs baseline (EIP-1559)
- Cost-saving analysis
- Continuous real-time inference mode

### Technical Details
- Python 3.10+ compatible
- PyTorch for LSTM implementation
- XGBoost for gradient boosting
- Web3.py for Ethereum interaction
- Pandas/NumPy for data processing
- YAML-based configuration
- CLI-based interface

### Documentation
- README with quick start guide
- SETUP guide with troubleshooting
- Inline code documentation (docstrings)
- Unit test examples
- PowerShell script examples

## [Future Roadmap]

### Planned Features
- [ ] Web dashboard for visualization
- [ ] Multi-network support (Mainnet, Arbitrum, Polygon)
- [ ] Advanced ensemble methods
- [ ] Transformer-based models
- [ ] Gas price prediction for different transaction types
- [ ] Alert system for optimal transaction timing
- [ ] Historical data caching
- [ ] Docker containerization
- [ ] REST API for predictions
- [ ] Grafana dashboard integration

### Research Extensions
- [ ] Uncertainty quantification
- [ ] Explainable AI features (SHAP values)
- [ ] Time series cross-validation
- [ ] Online learning capability
- [ ] Multi-step ahead prediction
- [ ] MEV-aware predictions

---

For detailed changes and commits, see Git history.
