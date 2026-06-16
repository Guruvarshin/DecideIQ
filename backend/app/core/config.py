from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str
    openai_api_key: str
    tavily_api_key: str

    mongodb_uri: str
    mongodb_db_name: str

    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 10080

    frontend_url: str = "http://localhost:3000"
    chroma_persist_path: str = "./chroma_data"

    class Config:
        env_file = ".env"


settings = Settings()
