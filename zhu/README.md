# Application package

This directory contains the FastAPI predictor (`backend_app.py`), its bounded
inverse-search engine (`inverse_design.py`), and the Rio web application
(`zhu/`). Start both services from the workspace root so they use the
same `.venv` and coordinated ports:

```bash
bash ../run.sh
```

The API returns resistivity in `μΩ·cm` with the invariant
`y = 10**y_log`. `POST /inverse-design` reuses that same prediction path to
rank a strictly bounded candidate grid; it does not load a second model or
write server-side result files. See the root [README](../README.md) for setup,
runtime configuration, and validation commands.
