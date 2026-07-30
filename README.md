# Sales Forecasting

![Sales Forecasting project cover](assets/recruiter/cover.png)

> **Portfolio lens:** A config-first foundation for turning retail history into a repeatable forecasting workflow.

This repository is the starting point for a sales-forecasting project. It contains the package layout, configuration files, a Flask entrypoint, a notebook for experiments, and the dependency set needed for the planned pipeline.

## Current state

The implementation is still at the foundation stage. `main.py` currently only initializes project logging; the model, data-ingestion, and training pipeline modules have not been implemented yet. This is useful as a project skeleton, but it is not a ready-to-run forecasting service.

## Layout

- `conf/config.yaml`, `params.yaml`, and `schema.yaml` hold the intended configuration, parameters, and input schema.
- `research/trials.ipynb` is the experimentation notebook.
- `src/` contains the package and future pipeline/model modules.
- `app.py` and `templates/` are reserved for the Flask interface.
- `Dockerfile` is the container starting point.

## Local setup

Use Python 3.10 or later.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

The final command verifies that the current package can start, but it does not produce a forecast. Add the data-processing and modelling stages before treating this as a deployable application.
