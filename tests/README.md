# Tests

Run all tests:
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

Run a single module:
```bash
pytest tests/test_ssi.py -v
```

## Test coverage targets

| Module | Tests |
|--------|-------|
| loader / validator | test_loader.py |
| feature_engineer | test_feature_engineer.py |
| temporal_splitter | test_temporal_splitter.py |
| adwin_detector | test_adwin.py |
| shap_engine | test_shap_engine.py |
| SSI + rank_shift | test_ssi.py |
