# ZTunnel - Development Documentation

> **Domain:** ztunnel.in  
> **Tagline:** "Instant tunnels. Zero config."  
> **Positioning:** ngrok alternative for India (INR pricing, low latency)

---

## 🎯 Product Vision

ZTunnel exposes local services to the internet through secure tunnels. A lightweight, fast tunneling solution with India-first pricing.

### Use Cases
1. **AI/ML Developers** - Expose Jupyter notebooks, Gradio demos, model APIs
2. **Web Developers** - Test webhooks, share localhost with clients
3. **IoT/Edge** - Access devices behind NAT
4. **API Testing** - Share local APIs with remote teams
5. **Demo Sharing** - Quick demos without deployment

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     ZTunnel Cloud                            │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                   Edge Server (Go)                      │ │
│  │  - WebSocket multiplexer                                │ │
│  │  - TLS termination                                      │ │
│  │  - Wildcard DNS: *.ztunnel.in                          │ │
│  │  - Rate limiting, auth                                  │ │
│  └─────────────────────────────────────────────────────────┘ │
│                           ↕ WebSocket                        │
└──────────────────────────────────────────────────────────────┘
                            │
                            │ Internet
                            │
┌──────────────────────────────────────────────────────────────┐
│                    Client (User's Machine)                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                   ZTunnel CLI (Go)                      │ │
│  │  $ ztunnel http 3000                                    │ │
│  │  → https://abc123.ztunnel.in                            │ │
│  └─────────────────────────────────────────────────────────┘ │
│                           ↕                                  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Local Service (port 3000)                  │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### Data Flow
1. Client CLI connects to edge server via WebSocket
2. Edge server assigns subdomain (e.g., `abc123.ztunnel.in`)
3. HTTP request hits edge → forwarded over WebSocket → CLI → local service
4. Response travels back the same path

---

## 💰 Pricing Tiers

| Feature | Free | Pro (₹299/mo) | Business (₹1499/mo) |
|---------|------|---------------|---------------------|
| Tunnels | 1 | 5 | Unlimited |
| Bandwidth | 1 GB/mo | 100 GB/mo | 1 TB/mo |
| Custom subdomain | ❌ | ✅ `myapp.ztunnel.in` | ✅ |
| Custom domain | ❌ | ❌ | ✅ `tunnel.yourdomain.com` |
| TCP tunnels | ❌ | ✅ | ✅ |
| Team members | 1 | 5 | Unlimited |
| Request inspection | Basic | Full | Full + Export |
| Uptime SLA | - | 99% | 99.9% |

---

## 🛠️ Tech Stack

### Edge Server (Go)
- **Why Go:** Fast compilation, excellent concurrency (goroutines), small binary, gorilla/websocket library
- Single binary deployment
- Handles thousands of concurrent connections

### CLI Client (Go)
- Single binary, cross-platform (Linux, macOS, Windows)
- No dependencies for end users
- Easy installation via curl/brew

### Backend API (Python FastAPI)
- User auth, billing, analytics
- API key management
- Razorpay integration for payments

### Infrastructure
- **Compute:** Hetzner/DigitalOcean (Mumbai region for low latency)
- **DNS:** Cloudflare (wildcard DNS + DDoS protection)
- **Database:** PostgreSQL
- **Cache:** Redis (sessions, rate limiting, tunnel registry)

---

## 📅 Development Phases

### Phase 1: Core Tunneling (Week 1-2)
- [ ] Edge server (Go)
  - [ ] WebSocket connection handler
  - [ ] Subdomain generation (random 8-char)
  - [ ] HTTP request forwarding
  - [ ] Connection pooling
- [ ] CLI client (Go)
  - [ ] `ztunnel http <port>` command
  - [ ] WebSocket connection to edge
  - [ ] Local HTTP forwarding
  - [ ] Reconnection logic
- [ ] Basic auth (API key)

### Phase 2: Production Ready (Week 3-4)
- [ ] TLS termination (Let's Encrypt wildcard)
- [ ] Wildcard DNS setup (`*.ztunnel.in`)
- [ ] Rate limiting (per tunnel, per user)
- [ ] Request/response logging
- [ ] Graceful shutdown
- [ ] Health checks
- [ ] Metrics endpoint

### Phase 3: User Management (Week 5-6)
- [ ] FastAPI backend
  - [ ] User signup/login (email + password)
  - [ ] API key generation/rotation
  - [ ] Usage tracking (bandwidth, requests)
- [ ] Dashboard (React/Next.js)
  - [ ] Active tunnels view
  - [ ] Request logs with filtering
  - [ ] Usage stats & graphs

### Phase 4: Monetization (Week 7-8)
- [ ] Razorpay integration
- [ ] Subscription plans (Free/Pro/Business)
- [ ] Custom subdomain reservation
- [ ] Bandwidth metering & alerts
- [ ] Invoice generation

### Phase 5: Advanced Features (Week 9+)
- [ ] TCP tunnels (SSH, databases, any port)
- [ ] Custom domains (CNAME setup)
- [ ] Request inspection/replay (like ngrok inspector)
- [ ] Team accounts & permissions
- [ ] Webhook notifications
- [ ] IP allowlisting

---

## 📁 Project Structure

```
ztunnel/
├── edge/                    # Edge server (Go)
│   ├── cmd/
│   │   └── edge/
│   │       └── main.go      # Entry point
│   ├── internal/
│   │   ├── server/
│   │   │   ├── server.go    # HTTP server setup
│   │   │   └── websocket.go # WS connection handler
│   │   ├── tunnel/
│   │   │   ├── manager.go   # Tunnel registry (map subdomain→conn)
│   │   │   └── tunnel.go    # Single tunnel struct
│   │   ├── proxy/
│   │   │   └── http.go      # HTTP request forwarding
│   │   └── auth/
│   │       └── auth.go      # API key validation
│   ├── go.mod
│   └── go.sum
│
├── cli/                     # CLI client (Go)
│   ├── cmd/
│   │   └── ztunnel/
│   │       └── main.go      # CLI entry point
│   ├── internal/
│   │   ├── tunnel/
│   │   │   └── client.go    # WebSocket client
│   │   ├── proxy/
│   │   │   └── local.go     # Local HTTP forwarding
│   │   └── config/
│   │       └── config.go    # Config file (~/.ztunnel/config.yaml)
│   ├── go.mod
│   └── go.sum
│
├── backend/                 # API backend (Python FastAPI)
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── core/
│   │   │   ├── config.py    # Settings
│   │   │   ├── database.py  # SQLAlchemy
│   │   │   └── security.py  # JWT, password hashing
│   │   ├── models/
│   │   │   ├── user.py
│   │   │   ├── api_key.py
│   │   │   ├── tunnel.py
│   │   │   └── subscription.py
│   │   ├── schemas/
│   │   │   ├── user.py
│   │   │   ├── auth.py
│   │   │   └── tunnel.py
│   │   ├── api/v1/
│   │   │   ├── router.py
│   │   │   └── endpoints/
│   │   │       ├── auth.py
│   │   │       ├── users.py
│   │   │       ├── tunnels.py
│   │   │       └── billing.py
│   │   └── services/
│   │       ├── user.py
│   │       ├── tunnel.py
│   │       └── billing.py
│   ├── requirements.txt
│   └── run.py
│
├── dashboard/               # Web dashboard (Next.js)
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   └── lib/
│   ├── package.json
│   └── next.config.js
│
├── deploy/
│   ├── docker-compose.yml
│   ├── Dockerfile.edge
│   ├── Dockerfile.backend
│   ├── nginx.conf           # For dashboard
│   └── systemd/
│       ├── ztunnel-edge.service
│       └── ztunnel-backend.service
│
└── docs/
    ├── quickstart.md
    ├── cli-reference.md
    ├── api-reference.md
    └── self-hosting.md
```

---

## 🔌 Wire Protocol (WebSocket)

### Client → Edge

```json
// 1. Register tunnel
{
  "type": "register",
  "api_key": "zt_live_xxxxxxxxxxxx",
  "protocol": "http",
  "subdomain": "myapp",        // optional (Pro/Business)
  "local_port": 3000
}

// 2. HTTP Response (after receiving request)
{
  "type": "http_response",
  "request_id": "req_abc123",
  "status": 200,
  "headers": {
    "Content-Type": "application/json",
    "X-Custom-Header": "value"
  },
  "body": "eyJzdWNjZXNzIjp0cnVlfQ=="  // base64 encoded
}

// 3. Ping (keepalive)
{
  "type": "ping"
}
```

### Edge → Client

```json
// 1. Tunnel registered successfully
{
  "type": "registered",
  "tunnel_id": "tun_xxxxxxxxxxxx",
  "url": "https://abc123.ztunnel.in",
  "expires_at": "2026-03-05T14:00:00Z"  // Free tier: 2 hours
}

// 2. Forward HTTP request to client
{
  "type": "http_request",
  "request_id": "req_abc123",
  "method": "POST",
  "path": "/api/chat",
  "query": "model=gpt4",
  "headers": {
    "Content-Type": "application/json",
    "Authorization": "Bearer xxx"
  },
  "body": "eyJtZXNzYWdlIjoiaGVsbG8ifQ=="  // base64 encoded
}

// 3. Pong (keepalive response)
{
  "type": "pong"
}

// 4. Error
{
  "type": "error",
  "code": "rate_limited",
  "message": "Too many requests"
}
```

---

## 🚀 CLI Commands

```bash
# Authentication
$ ztunnel auth login              # Login with email/password
$ ztunnel auth logout             # Clear saved credentials
$ ztunnel auth status             # Show current user & plan

# HTTP Tunnels
$ ztunnel http 3000               # Expose port 3000
$ ztunnel http 3000 --subdomain myapp   # Custom subdomain (Pro)
$ ztunnel http 8080 --inspect     # Enable request inspector

# TCP Tunnels (Pro/Business)
$ ztunnel tcp 22                  # Expose SSH
$ ztunnel tcp 5432                # Expose PostgreSQL

# Management
$ ztunnel list                    # List active tunnels
$ ztunnel stop <tunnel_id>        # Stop a tunnel
$ ztunnel logs <tunnel_id>        # View tunnel logs

# Configuration
$ ztunnel config set api_key zt_xxx   # Set API key
$ ztunnel config get                   # Show config
```

### CLI Output Example

```
$ ztunnel http 3000

   ╔══════════════════════════════════════════════════════╗
   ║                                                      ║
   ║   ZTunnel v1.0.0                                     ║
   ║                                                      ║
   ║   Tunnel Status: online                              ║
   ║   Forwarding:    https://k8f2m9x1.ztunnel.in        ║
   ║                  → http://localhost:3000             ║
   ║                                                      ║
   ║   Inspect:       http://localhost:4040               ║
   ║                                                      ║
   ╚══════════════════════════════════════════════════════╝

   Connections:

   POST /api/chat  200 OK  142ms
   GET  /health    200 OK   12ms
   POST /api/chat  200 OK  156ms
```

---

## 🔒 Security

1. **TLS Everywhere** - All tunnel traffic encrypted (TLS 1.3)
2. **API Key Rotation** - Users can revoke/regenerate keys
3. **Rate Limiting** - Per tunnel and per user limits
4. **Request Size Limits** - Max 10MB request body (configurable)
5. **Tunnel Timeout** - Free tier: 2 hour max session
6. **IP Allowlisting** - Business tier: restrict tunnel access
7. **Audit Logs** - All API actions logged

### API Key Format
```
zt_live_xxxxxxxxxxxxxxxxxxxx    # Production key
zt_test_xxxxxxxxxxxxxxxxxxxx    # Test key (no billing)
```

---

## 📊 Database Schema

```sql
-- Users
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    plan VARCHAR(20) DEFAULT 'free',  -- free, pro, business
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- API Keys
CREATE TABLE api_keys (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    key_hash VARCHAR(255) NOT NULL,      -- hashed API key
    key_prefix VARCHAR(20) NOT NULL,     -- zt_live_xxxx (for display)
    name VARCHAR(100),                   -- user-defined name
    last_used_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    revoked_at TIMESTAMP
);

-- Tunnels (active sessions)
CREATE TABLE tunnels (
    id VARCHAR(50) PRIMARY KEY,          -- tun_xxxxxxxxxxxx
    user_id INTEGER REFERENCES users(id),
    subdomain VARCHAR(50) NOT NULL,
    protocol VARCHAR(10) NOT NULL,       -- http, tcp
    local_port INTEGER NOT NULL,
    edge_server VARCHAR(100),            -- which edge server
    started_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    bytes_in BIGINT DEFAULT 0,
    bytes_out BIGINT DEFAULT 0,
    requests_count INTEGER DEFAULT 0
);

-- Reserved Subdomains (Pro/Business)
CREATE TABLE reserved_subdomains (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    subdomain VARCHAR(50) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Usage (daily aggregates)
CREATE TABLE usage (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    date DATE NOT NULL,
    bytes_in BIGINT DEFAULT 0,
    bytes_out BIGINT DEFAULT 0,
    requests_count INTEGER DEFAULT 0,
    tunnel_minutes INTEGER DEFAULT 0,
    UNIQUE(user_id, date)
);

-- Subscriptions
CREATE TABLE subscriptions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    plan VARCHAR(20) NOT NULL,
    razorpay_subscription_id VARCHAR(100),
    status VARCHAR(20),                  -- active, cancelled, past_due
    current_period_start TIMESTAMP,
    current_period_end TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 🌐 API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/signup` | Create account |
| POST | `/api/v1/auth/login` | Get JWT token |
| POST | `/api/v1/auth/logout` | Logout |
| GET | `/api/v1/auth/me` | Current user |

### API Keys
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/keys` | List API keys |
| POST | `/api/v1/keys` | Create new key |
| DELETE | `/api/v1/keys/{id}` | Revoke key |

### Tunnels
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/tunnels` | List active tunnels |
| GET | `/api/v1/tunnels/{id}` | Tunnel details |
| DELETE | `/api/v1/tunnels/{id}` | Stop tunnel |
| GET | `/api/v1/tunnels/{id}/logs` | Request logs |

### Subdomains (Pro/Business)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/subdomains` | List reserved |
| POST | `/api/v1/subdomains` | Reserve subdomain |
| DELETE | `/api/v1/subdomains/{subdomain}` | Release |

### Usage & Billing
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/usage` | Usage stats |
| GET | `/api/v1/billing/subscription` | Current plan |
| POST | `/api/v1/billing/subscribe` | Subscribe to plan |
| POST | `/api/v1/billing/cancel` | Cancel subscription |

### Edge Server (Internal)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/internal/validate-key` | Validate API key |
| POST | `/internal/tunnel-started` | Report tunnel start |
| POST | `/internal/tunnel-stopped` | Report tunnel end |
| POST | `/internal/usage` | Report usage metrics |

---

## 🏁 Getting Started (Development)

### Prerequisites
- Go 1.21+
- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Node.js 20+ (for dashboard)

### 1. Clone & Setup

```bash
# Create project directory
mkdir ztunnel && cd ztunnel

# Initialize edge server
mkdir -p edge/cmd/edge edge/internal/{server,tunnel,proxy,auth}
cd edge && go mod init github.com/zyora/ztunnel-edge
go get github.com/gorilla/websocket
go get github.com/rs/zerolog

# Initialize CLI
cd .. && mkdir -p cli/cmd/ztunnel cli/internal/{tunnel,proxy,config}
cd cli && go mod init github.com/zyora/ztunnel-cli
go get github.com/spf13/cobra
go get github.com/gorilla/websocket

# Initialize backend
cd .. && mkdir -p backend/app/{core,models,schemas,api/v1/endpoints,services}
cd backend
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn sqlalchemy asyncpg python-jose passlib bcrypt
```

### 2. Environment Variables

```bash
# backend/.env
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/ztunnel
REDIS_URL=redis://localhost:6379/0
JWT_SECRET=your-super-secret-key
RAZORPAY_KEY_ID=rzp_test_xxx
RAZORPAY_KEY_SECRET=xxx

# edge/.env
BACKEND_URL=http://localhost:8000
EDGE_DOMAIN=ztunnel.in
REDIS_URL=redis://localhost:6379/0
```

### 3. Run Locally

```bash
# Terminal 1: Backend
cd backend && python run.py

# Terminal 2: Edge server
cd edge && go run cmd/edge/main.go

# Terminal 3: Test with CLI
cd cli && go run cmd/ztunnel/main.go http 3000
```

---

## 📊 Metrics & Monitoring

### Prometheus Metrics (Edge Server)
```
ztunnel_active_tunnels           # Gauge: current active tunnels
ztunnel_requests_total           # Counter: total requests forwarded
ztunnel_request_duration_seconds # Histogram: request latency
ztunnel_bytes_transferred_total  # Counter: bytes in/out
ztunnel_websocket_connections    # Gauge: active WebSocket connections
ztunnel_errors_total             # Counter: errors by type
```

### Health Endpoints
```
GET /health          # Basic health check
GET /health/ready    # Ready for traffic
GET /metrics         # Prometheus metrics
```

---

## 🚢 Deployment

### Docker Compose (Development)

```yaml
version: '3.8'
services:
  edge:
    build:
      context: ./edge
      dockerfile: Dockerfile
    ports:
      - "80:80"
      - "443:443"
    environment:
      - BACKEND_URL=http://backend:8000
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
      - backend

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:password@db:5432/ztunnel
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=ztunnel
      - POSTGRES_PASSWORD=password
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redisdata:/data

volumes:
  pgdata:
  redisdata:
```

### Production Checklist
- [ ] Wildcard SSL certificate (`*.ztunnel.in`)
- [ ] Cloudflare DNS with wildcard record
- [ ] Rate limiting in nginx/edge
- [ ] PostgreSQL with replicas
- [ ] Redis cluster for HA
- [ ] Prometheus + Grafana monitoring
- [ ] Log aggregation (Loki/ELK)
- [ ] Automated backups

---

## 📝 Notes

- Start with HTTP tunnels only (simpler to implement)
- TCP tunnels need port allocation strategy (range: 40000-50000)
- Consider WebRTC for P2P in future (lower latency)
- Mobile app for tunnel management (future roadmap)
- Self-hosting option for enterprise (future)

---

*Last updated: March 2026*
