# Zeta AI — Deployment Guide
**Domain:** https://www.zetaai.com  
**Server IP:** 150.230.133.117  
**Stack:** Flask · MongoDB · Pinecone · Gemini · OpenAI

---

## File Structure

```
/opt/zetaai/
├── app.py          # Flask backend (entry point)
├── db.py           # MongoDB data layer
├── mars_ai.py      # Bridge to AI engine
├── proper_ai.py    # Core AI engine (Pinecone + Gemini + OpenAI)
├── requirements.txt
├── .env            # Secrets — never commit this
└── index.html      # Frontend (served by Flask at /)
```

---

## 1 — Server Setup (Ubuntu 22.04)

```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip nginx

# MongoDB
curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | sudo gpg --dearmor -o /etc/apt/keyrings/mongodb-server-7.0.gpg
echo "deb [ arch=amd64 signed-by=/etc/apt/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
sudo apt update && sudo apt install -y mongodb-org
sudo systemctl enable mongod && sudo systemctl start mongod
```

---

## 2 — App Install

```bash
sudo mkdir -p /opt/zetaai
sudo chown $USER:$USER /opt/zetaai
cd /opt/zetaai

# Copy all files here, then:
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Copy .env and fill in real secrets
cp .env.example .env
nano .env
```

---

## 3 — Gunicorn Systemd Service

Create `/etc/systemd/system/zetaai.service`:

```ini
[Unit]
Description=Zeta AI Flask App
After=network.target mongod.service

[Service]
User=www-data
WorkingDirectory=/opt/zetaai
EnvironmentFile=/opt/zetaai/.env
ExecStart=/opt/zetaai/venv/bin/gunicorn \
    --workers 4 \
    --bind 127.0.0.1:5000 \
    --timeout 120 \
    app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable zetaai
sudo systemctl start zetaai
```

---

## 4 — Nginx Reverse Proxy + HTTPS

`/etc/nginx/sites-available/zetaai`:

```nginx
server {
    listen 80;
    server_name zetaai.com www.zetaai.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name zetaai.com www.zetaai.com;

    ssl_certificate     /etc/letsencrypt/live/zetaai.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/zetaai.com/privkey.pem;

    client_max_body_size 20M;

    location / {
        proxy_pass         http://127.0.0.1:5000;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/zetaai /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# Free SSL via Certbot
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d zetaai.com -d www.zetaai.com
```

---

## 5 — DNS Records

Point these at `150.230.133.117` in your DNS panel:

| Type | Name | Value             |
|------|------|-------------------|
| A    | @    | 150.230.133.117   |
| A    | www  | 150.230.133.117   |

---

## 6 — Pinecone Setup

The app auto-creates the `zeta-memory` index (384-dim, cosine) on first run.  
Verify at https://app.pinecone.io after first chat request.

---

## 7 — Health Check

```bash
curl https://www.zetaai.com/api/health
# Expected: {"status":"ok","engine":{"status":"online"},"db":"mongodb","vectors":"pinecone"}
```

---

## Key Changes from v1

| Component     | Before       | After           |
|---------------|--------------|-----------------|
| Database      | Oracle DB    | MongoDB         |
| Vector search | FAISS (local)| Pinecone (cloud)|
| Paths         | `d:\mars`    | `/opt/zetaai`   |
| CORS origins  | `*`          | zetaai.com only |
| Process mgr   | —            | Gunicorn + Nginx|
