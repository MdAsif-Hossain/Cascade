# Deployment notes

## Backend — Render

The blueprint at `../render.yaml` describes both services. Two things need doing by hand:

1. **Set the secrets** in the Render dashboard: `GROQ_API_KEY`, `GOOGLE_AI_STUDIO_API_KEY`, `OPENROUTER_API_KEY`, and `DATABASE_URL` (the Supabase Postgres connection string). They are marked `sync: false` in the blueprint so they are never committed.
2. **Set `TARGET_URL`** on the `cascade-keepalive` cron service to the deployed API's base URL, once Render has assigned one.

### Why the keep-alive exists

Render's free web services are spun down after 15 minutes of inactivity, and the next request pays a cold start of roughly 50 seconds. To a student that is indistinguishable from a broken site. The cron service pings `/health` every 10 minutes to keep the container warm.

`/health` is deliberately cheap — it makes no provider or database call — so the ping cannot itself consume free-tier quota or make a rate-limited provider look like a dead service.

### Memory

The free tier gives 512 MB. The image runs a single uvicorn worker because two copies of the interpreter plus scikit-learn will not fit. This is not a throughput problem: the workload is I/O-bound waiting on provider calls, which one async worker handles well.

## Frontend — Vercel

Set `NEXT_PUBLIC_API_BASE` to the Render URL in the Vercel project settings. Without it the frontend points at `http://127.0.0.1:8000` and every request fails in production.

## Database

SQLite locally (`sqlite:///./cascade.db`, created automatically). Supabase Postgres in production. The schema is created on startup by `create_tables()`; Alembic owns any subsequent migration.

**Never store files on Render's disk** — it is ephemeral and wiped on every deploy. The only files the backend reads are `artifacts/classifier.joblib` and `artifacts/metadata.json`, which are baked into the image at build time.
