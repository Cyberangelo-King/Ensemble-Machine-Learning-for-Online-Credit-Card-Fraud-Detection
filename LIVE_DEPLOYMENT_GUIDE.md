# Demo Deployment Guide

This repository can be published as a **research demonstration** with a FastAPI service and a static React dashboard. It is not connected to a bank, issuer, acquirer, payment processor or live card network.

## Recommended: Render Blueprint

1. Push the `workbench/model-audit-hardening-2026-11` branch to GitHub for review. Keep the PR unmerged until the research changes and generated artifacts have been reviewed.
2. Upload the intended model bundle to a trusted HTTPS artifact store.
3. Calculate the model digest:
   ```bash
   sha256sum results/stacking_model.pkl
   ```
4. In Render, select **New + → Blueprint** and connect this repository.
5. Supply `MODEL_URL` and `MODEL_SHA256` when Render requests the unsynchronised environment values.
6. Confirm that the backend health endpoint returns `status=ok` and `model_loaded=true`.
7. Open the static frontend and start the simulated transaction replay.

The Blueprint's frontend is a static Vite build. The previous Node 18 service/start command has been removed; this frontend uses Vite 8 and a current Node runtime.

## Local demonstration

Follow [DEPLOYMENT.md](DEPLOYMENT.md) for the full local setup, environment variables, synthetic demo generation, API checks, and benchmark instructions.

## Security boundary

The public Blueprint uses `APP_ENV=demo`. Do not upload customer information, cardholder data, payment credentials or private transaction streams. Do not place secrets in `VITE_*` frontend variables. The browser app does not yet implement end-user authentication or a secure API proxy. This deployment must not be described as production-ready.

The simulated WebSocket feed replays sample rows from the backend's local dataset or generated synthetic rows. It is not a real-time fraud detection integration.
