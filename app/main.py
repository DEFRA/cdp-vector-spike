import os
from contextlib import asynccontextmanager
from logging import getLogger

import uvicorn
from fastapi import FastAPI

from app.common.mongo import get_mongo_client
from app.common.tracing import TraceIdMiddleware
from app.config import config
from app.health.router import router as health_router
from app.vector.router import router as vector_router

logger = getLogger(__name__)


app = FastAPI()

# Setup middleware
app.add_middleware(TraceIdMiddleware)

# Setup Routes
app.include_router(health_router)
app.include_router(vector_router)


def main() -> None:  # pragma: no cover
    if config.http_proxy:
        os.environ["HTTP_PROXY"] = str(config.http_proxy)
        os.environ["HTTPS_PROXY"] = str(config.http_proxy)
    else:
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
    
    logger.info("starting server on %s", config.port)

    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        log_config=config.log_config,
        reload=config.python_env == "development",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
