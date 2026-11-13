from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import UPLOAD_DIR
from .ingestion.router import router as ingestion_router
from .search.router import router as search_router

app = FastAPI(title="LectureLens")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(ingestion_router)
app.include_router(search_router)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=UPLOAD_DIR), name="media")
