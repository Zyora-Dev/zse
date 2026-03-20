# ZMesh Progress - Saved State

> **Domain:** zmesh.in  
> **Last Updated:** March 5, 2026

---

## ✅ Completed

### Backend Structure
```
zmesh/backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py      # Settings (JWT_SECRET, DB URL, etc.)
│   │   ├── database.py    # SQLAlchemy async setup
│   │   └── security.py    # JWT + password hashing
│   ├── models/
│   │   ├── __init__.py
│   │   └── user.py        # User SQLAlchemy model
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── auth.py        # LoginRequest, TokenResponse
│   │   ├── common.py      # MessageResponse
│   │   └── user.py        # UserCreate, UserResponse, etc.
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── router.py
│   │       └── endpoints/
│   │           ├── __init__.py
│   │           ├── auth.py    # signup, login, me, logout
│   │           └── users.py   # profile CRUD
│   └── services/
│       ├── __init__.py
│       └── user.py        # Business logic
├── requirements.txt
└── run.py
```

### Database
- **PostgreSQL database:** `zyoramesh`
- **Users table:** 24 columns with full profile fields

```sql
-- User table columns
id, email, username, hashed_password, full_name, bio, avatar_url,
website, github_url, twitter_url, linkedin_url, organization,
job_title, country, timezone, use_case, is_active, is_verified,
is_admin, wallet_balance, created_at, updated_at, last_login, 
refresh_token
```

### Auth Endpoints (Working)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/auth/signup` | POST | Create account |
| `/api/v1/auth/login` | POST | Get JWT token |
| `/api/v1/auth/me` | GET | Current user profile |
| `/api/v1/auth/logout` | POST | Logout (clears refresh) |

### Profile Endpoints (Working)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/users/me` | GET | Full profile |
| `/api/v1/users/me` | PATCH | Update profile |
| `/api/v1/users/{username}` | GET | Public profile |

### Dependencies (requirements.txt)
```
fastapi==0.109.0
uvicorn[standard]==0.27.0
sqlalchemy==2.0.25
asyncpg==0.29.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
bcrypt==4.0.1  # Downgraded for passlib compatibility
pydantic==2.5.3
pydantic-settings==2.1.0
python-multipart==0.0.6
```

### Bug Fixes Applied
1. **bcrypt 5.x incompatibility** → Downgraded to bcrypt 4.0.1
2. **JWT "sub must be string"** → Convert user_id to string in token

---

## 🔄 In Progress / Planned

### Next: GPU Provider Module
```python
# Planned structure
app/
├── services/
│   ├── gpu_providers/
│   │   ├── __init__.py
│   │   ├── base.py        # Abstract provider
│   │   ├── runpod.py      # RunPod API
│   │   └── vastai.py      # Vast.ai API
│   └── gpu_rental.py      # Rental orchestration
├── models/
│   ├── gpu.py             # GPU instance model
│   └── rental.py          # Rental transaction
└── api/v1/endpoints/
    └── gpus.py            # GPU listing, rental endpoints
```

### Planned Features
- [ ] GPU listing (fetch from RunPod/Vast.ai)
- [ ] Rental flow (select GPU → pay → provision)
- [ ] Wallet system (INR balance, top-up)
- [ ] Razorpay integration
- [ ] ZTunnel integration (auto-tunnel for GPU access)
- [ ] ZSE integration (one-click fine-tuning)

---

## 🔧 Configuration

### Environment Variables
```bash
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/zyoramesh
JWT_SECRET=your-secret-key-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### Server
- **Port:** 7001
- **Command:** `cd zmesh/backend && python run.py`

---

## 📊 Business Model

### GPU Sourcing
- RunPod API → Proxy URLs: `https://{pod_id}-{port}.proxy.runpod.net`
- Vast.ai API → SSH access

### Pricing Strategy
- Add 10-20% margin on cloud GPU costs
- Bill in INR (India-first)
- Hourly billing, prepaid wallet

### Integration with Other Products
| Product | Integration |
|---------|-------------|
| **ZSE** | One-click fine-tuning on rented GPUs |
| **ZTunnel** | Access GPUs via `gpu-xxx.ztunnel.in` |

---

## 📁 File Locations

| File | Purpose |
|------|---------|
| `/Users/redfoxhotels/zse/zmesh/backend/app/main.py` | FastAPI app |
| `/Users/redfoxhotels/zse/zmesh/backend/app/core/config.py` | Settings |
| `/Users/redfoxhotels/zse/zmesh/backend/app/models/user.py` | User model |
| `/Users/redfoxhotels/zse/zmesh/backend/app/api/v1/endpoints/auth.py` | Auth routes |

---

## 🧪 Testing

```bash
# Signup
curl -X POST http://localhost:7001/api/v1/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","username":"testuser","password":"password123"}'

# Login
curl -X POST http://localhost:7001/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123"}'

# Get profile (with token)
curl http://localhost:7001/api/v1/auth/me \
  -H "Authorization: Bearer <token>"
```

---

## 📌 Resume Points

When continuing ZMesh development:

1. **Immediate:** GPU provider module (RunPod API integration)
2. **Then:** GPU listing endpoint `/api/v1/gpus`
3. **Then:** Rental flow with wallet deduction
4. **Then:** Razorpay for wallet top-up
5. **Finally:** ZTunnel integration for GPU access

---

*Saved: March 5, 2026*
