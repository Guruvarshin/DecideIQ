from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.database import connect_to_mongo, close_mongo

# LangSmith tracing is enabled automatically when LANGSMITH_API_KEY and
# LANGSMITH_TRACING=true are set in the environment — no code needed here.
from app.api.auth import router as auth_router
from app.api.sessions import router as sessions_router
from app.api.documents import router as documents_router
from app.api.questions import router as questions_router
from app.api.comparison import router as comparison_router
from app.api.evaluation import router as evaluation_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mongo()
    yield
    await close_mongo()


app = FastAPI(title="DecideIQ API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(sessions_router)
app.include_router(documents_router)
app.include_router(questions_router)
app.include_router(comparison_router)
app.include_router(evaluation_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "decideiq-backend"}
