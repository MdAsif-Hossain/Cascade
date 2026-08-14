# Minimal keep-alive pinger. Render's free web services sleep after 15 minutes
# idle; this hits /health every 10 minutes so students never meet a cold start.
FROM alpine:3.20
RUN apk add --no-cache curl
CMD ["sh", "-c", "curl -fsS --max-time 30 \"$TARGET_URL/health\" && echo ' ok'"]
