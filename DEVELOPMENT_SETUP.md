# DataForge: Deep Learning-Based Synthetic Data Generation
## Phase 3: Development Environment Setup & Verification Document
**Course / Subject:** AI23531 – Deep Learning  
**Project Title:** DataForge – Deep Learning-Based Synthetic Data Generation Platform  
**Document Version:** 1.0.0  
**Status:** Operational & Verified  

---

## 1. Environment & Runtime Specifications

| Component | Technology | Version | Location / Path |
| :--- | :--- | :--- | :--- |
| **Node.js Runtime** | Node.js (LTS) | `v20.18.0` | `AppData\Local\Programs\node-v20.18.0-win-x64\node.exe` |
| **Package Manager** | NPM | `10.8.2` | `AppData\Local\Programs\node-v20.18.0-win-x64\npm.cmd` |
| **Frontend Framework** | React + TypeScript | `18.3.1` | `frontend/` (Vite 5.4.11) |
| **Styling & Charts** | Tailwind CSS + Recharts | `3.4.15` / `2.13.3` | `frontend/` |
| **Icons** | Lucide React | `0.460.0` | `frontend/` |
| **Python Runtime** | Python 64-bit | `3.15.0` | `AppData\Local\Python\pythoncore-3.15-64\python.exe` |
| **Backend API Gateway**| FastAPI + Uvicorn | `0.143.0` / `0.54.0` | `backend/app/main.py` |
| **Data Processing** | NumPy + Pandas | `2.5.4` / `3.0.6` | `backend/` |
| **Statistical Engine** | SciPy + Scikit-Learn | `1.18.1` / `1.9.1` | `backend/` |
| **Database** | SQLite3 | Embedded | `backend/dataforge.db` |

---

## 2. Directory Structure

```
c:\Users\divya\OneDrive\Desktop\dl project\
├── PROJECT_REQUIREMENTS_DOCUMENT.md
├── SYSTEM_DESIGN_DOCUMENT.md
├── DEVELOPMENT_SETUP.md
├── backend\
│   ├── .env
│   ├── requirements.txt
│   ├── app\
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── api\
│   │   ├── core\
│   │   ├── models\
│   │   └── services\
│   └── data\
│       ├── uploads\
│       ├── models\
│       └── synthetic\
└── frontend\
    ├── .env
    ├── package.json
    ├── tsconfig.json
    ├── vite.config.ts
    ├── tailwind.config.js
    ├── postcss.config.js
    ├── index.html
    └── src\
        ├── App.tsx
        ├── main.tsx
        ├── index.css
        ├── vite-env.d.ts
        ├── components\
        ├── pages\
        ├── services\
        └── types\
```

---

## 3. Environment Variables Configuration

### 3.1 Backend Configuration (`backend/.env`)
```ini
PROJECT_NAME="DataForge Deep Learning Synthetic Data Generator"
ENVIRONMENT="development"
HOST="127.0.0.1"
PORT=8000
DATABASE_URL="sqlite:///./dataforge.db"
UPLOAD_DIR="./data/uploads"
MODEL_DIR="./data/models"
SYNTHETIC_DIR="./data/synthetic"
CORS_ORIGINS="http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"
```

### 3.2 Frontend Configuration (`frontend/.env`)
```ini
VITE_API_BASE_URL="http://127.0.0.1:8000"
VITE_APP_TITLE="DataForge - Deep Learning Synthetic Data Generator"
```

---

## 4. Verification & Communication Tests

### 4.1 Backend Direct Health Check
- **Endpoint:** `GET http://127.0.0.1:8000/api/health`
- **Response:**
```json
{
  "status": "online",
  "message": "DataForge Backend is operational and ready for requests",
  "version": "1.0.0",
  "deep_learning_engine": "VAE Neural Engine (Active)"
}
```

### 4.2 Frontend Reverse Proxy Verification
- **Endpoint:** `GET http://127.0.0.1:5173/api/health` (routed through Vite dev server)
- **Status:** `200 OK`
- **Result:** Frontend successfully proxies requests to the backend without CORS restrictions.

### 4.3 TypeScript Production Build Test
- **Command:** `npm run build`
- **Output:**
```
✓ 1576 modules transformed.
dist/index.html                   0.55 kB
dist/assets/index-oyvS5-FC.css   14.16 kB
dist/assets/index-CjahLxEC.js   159.99 kB
✓ built in 24.58s
```

---

## 5. Development Launch Commands

### Running Backend API:
```powershell
cd "c:\Users\divya\OneDrive\Desktop\dl project\backend"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Running Frontend UI:
```powershell
cd "c:\Users\divya\OneDrive\Desktop\dl project\frontend"
npm run dev -- --host 127.0.0.1 --port 5173
```
Access the application at: **http://127.0.0.1:5173/**
API Swagger documentation at: **http://127.0.0.1:8000/docs**
