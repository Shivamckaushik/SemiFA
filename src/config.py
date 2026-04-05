"""Centralised settings loaded from environment / .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM / Model
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    hf_token: str = Field(default="", alias="HF_TOKEN")
    llava_model_id: str = Field(
        default="llava-hf/llava-v1.6-mistral-7b-hf", alias="LLAVA_MODEL_ID"
    )
    llava_finetuned_path: str = Field(
        default="./models/llava-semiconductor-qlora", alias="LLAVA_FINETUNED_PATH"
    )
    dinov2_model_id: str = Field(
        default="facebook/dinov2-base", alias="DINOV2_MODEL_ID"
    )

    # Qdrant
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_collection: str = Field(
        default="semiconductor_defects", alias="QDRANT_COLLECTION"
    )

    # MinIO
    minio_endpoint: str = Field(default="localhost:9000", alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", alias="MINIO_SECRET_KEY")
    minio_bucket: str = Field(default="inspection-images", alias="MINIO_BUCKET")
    minio_secure: bool = Field(default=False, alias="MINIO_SECURE")

    # TimescaleDB
    timescale_host: str = Field(default="localhost", alias="TIMESCALE_HOST")
    timescale_port: int = Field(default=5432, alias="TIMESCALE_PORT")
    timescale_db: str = Field(
        default="semiconductor_telemetry", alias="TIMESCALE_DB"
    )
    timescale_user: str = Field(default="postgres", alias="TIMESCALE_USER")
    timescale_password: str = Field(default="postgres", alias="TIMESCALE_PASSWORD")

    @property
    def timescale_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.timescale_user}:{self.timescale_password}"
            f"@{self.timescale_host}:{self.timescale_port}/{self.timescale_db}"
        )

    # MQTT
    mqtt_broker_host: str = Field(default="localhost", alias="MQTT_BROKER_HOST")
    mqtt_broker_port: int = Field(default=1883, alias="MQTT_BROKER_PORT")
    mqtt_username: str = Field(default="", alias="MQTT_USERNAME")
    mqtt_password: str = Field(default="", alias="MQTT_PASSWORD")
    mqtt_equipment_topic: str = Field(
        default="fab/equipment/#", alias="MQTT_EQUIPMENT_TOPIC"
    )

    # SECS/GEM
    secsgem_host: str = Field(default="127.0.0.1", alias="SECSGEM_HOST")
    secsgem_port: int = Field(default=5000, alias="SECSGEM_PORT")
    secsgem_session_id: int = Field(default=1, alias="SECSGEM_SESSION_ID")

    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # Reports
    report_output_dir: str = Field(
        default="./reports/output", alias="REPORT_OUTPUT_DIR"
    )


settings = Settings()
