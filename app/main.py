from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db import Base, engine
import app.models  # noqa: F401 - ensure models are registered
from app.routers.chat import router as chat_router
from app.routers.plans import router as plans_router
from app.routers.runs import router as runs_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="retail-data-workbench", lifespan=lifespan)

# Allow the Vite dev server (http://localhost:5173) to call the backend
# directly during local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs_router)
app.include_router(plans_router)
app.include_router(chat_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
