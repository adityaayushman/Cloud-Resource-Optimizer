# Deployment

Repository: **https://github.com/adityaayushman/Cloud-Resource-Optimizer**

Backend on Render, frontend on Vercel, both deploying from the same repository
and the same `main` branch.

---

## 1. Push the repository

```bash
cd "d:/projects/cloud resourse optimizer"
git push -u origin main
```

If prompted, authenticate with a GitHub personal access token (Settings →
Developer settings → Personal access tokens → Fine-grained, `Contents: Read and
write` on this repository). Alternatively `gh auth login` first, then push.

---

## 2. Backend — Render

The repository root contains `render.yaml`, so Render can configure the service
itself.

1. https://dashboard.render.com → **New** → **Blueprint**
2. Connect the `Cloud-Resource-Optimizer` repository
3. Render reads `render.yaml` and proposes a web service named
   `cloud-resource-optimizer-api`. Apply it.
4. Wait for the first build (**5–8 minutes** — it generates the dataset and
   trains every model artifact; the log ends with the ablation table).
5. Note the service URL, e.g. `https://cloud-resource-optimizer-api.onrender.com`

Verify:

```bash
curl https://<your-service>.onrender.com/api/health
```

`artifacts_ready` must be `true`. If it is `false`, the build's training step
failed — check the Render build log for the `scripts/train.py` output.

### Why the build trains the models

`.gitignore` excludes `backend/artifacts/*.joblib` and the generated dataset.
Committing ~35 MB of binary model files would bloat the repository and, worse,
allow a stale model to be served by newer code. Building them during deploy costs
a few minutes and guarantees artifacts and code always match.

---

## 3. Frontend — Vercel

1. https://vercel.com/new → import the `Cloud-Resource-Optimizer` repository
2. **Root Directory:** `frontend` ← this matters; the repo is a monorepo
3. Framework preset: **Vite** (build `npm run build`, output `dist` — already in
   `frontend/vercel.json`)
4. **Environment Variables** → add:

   | Name | Value |
   |---|---|
   | `VITE_API_BASE_URL` | `https://<your-render-service>.onrender.com` |

   Vite inlines `VITE_*` variables at **build** time, so changing this requires a
   redeploy, not just a restart.
5. Deploy.

---

## 4. Close the CORS loop

Back in Render → your service → **Environment** → add:

| Name | Value |
|---|---|
| `CORS_ORIGINS` | `https://<your-project>.vercel.app` |

Save; Render restarts automatically. Preview deployments on `*.vercel.app` are
already permitted by a regex in `app/main.py`, so only the production origin
needs listing here. Multiple origins are comma-separated.

---

## 5. Verify end to end

1. Open the Vercel URL.
2. Press **Initialise engine**. The boot log should reach "Session ready".
3. Press **Step** a few times — the demand/forecast/capacity chart fills in and
   the engine log shows DQN actions.
4. **Prediction & XAI** → **Predict** → the SHAP bars render.
5. **Results & Ablation** → the measured comparison table loads.

### If the ignition screen reports it cannot reach the API

| Symptom | Cause | Fix |
|---|---|---|
| Times out on first try, works on retry | Render free instance was asleep | Expected — it needs 30–60 s to wake |
| "Cannot reach the API at http://localhost:8000" | `VITE_API_BASE_URL` was not set at build time | Set it in Vercel and **redeploy** |
| Browser console shows a CORS error | `CORS_ORIGINS` missing or wrong on Render | Set it to the exact Vercel origin, no trailing slash |
| `artifacts_ready: false` | Training failed during the Render build | Check the build log; usually the free tier ran out of memory — lower `--rl-episodes` in `render.yaml` |

---

## Free-tier characteristics

These are properties of the hosting, not bugs — worth knowing before a demo:

- **Cold starts.** The API sleeps after ~15 minutes idle. Open it a minute before
  demonstrating.
- **Ephemeral sessions.** Simulation state lives in memory (`app/sessions.py`) and
  is lost on restart. The dashboard creates a fresh session automatically.
- **Single worker.** `render.yaml` sets `--workers 1` deliberately: sessions are
  in-process, so a second worker would serve requests from a fleet the first
  worker does not have.
- **Build minutes.** Every push to `main` triggers a rebuild including training.
  Set `autoDeploy: false` in `render.yaml` if that becomes inconvenient.

---

## Running locally

```bash
# terminal 1
cd backend
pip install -r requirements-dev.txt
python scripts/generate_data.py --days 30 --interval 5
python scripts/train.py
python scripts/evaluate.py
uvicorn app.main:app --reload

# terminal 2
cd frontend
npm install
cp .env.example .env
npm run dev
```

Backend on http://localhost:8000 (docs at `/docs`), dashboard on
http://localhost:5173. The default `CORS_ORIGINS` already allows the Vite dev
server.
