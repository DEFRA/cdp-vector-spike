from logging import getLogger

from fastapi import APIRouter, BackgroundTasks
from ingestion import ensure_index, s3vectors, store_embeddings
from pydantic import BaseModel

from app.config import config

router = APIRouter(prefix="/vector")
logger = getLogger(__name__)


class Document(BaseModel):
    filename: str
    text: str


def task_ensure_index():
    try:
        logger.info("started ensure index task")
        ensure_index()
        logger.info("ended ensure index task")
    except Exception as e:
        logger.error(e)


def task_embed(doc: Document):
    try:
        logger.info("started store_embeddings task")
        store_embeddings(doc.filename, doc.text)
        logger.info("ended store_embeddings task")
    except Exception as e:
        logger.error(e)


@router.get("/index")
async def index(background_tasks: BackgroundTasks):
    logger.info("index endpoint")
    background_tasks.add_task(task_ensure_index)
    return {"ok": True}


# basic endpoint example
@router.post("/embed")
async def embed(background_tasks: BackgroundTasks):
    logger.info("index endpoint")
    background_tasks.add_task(task_ensure_index)
    return {"ok": True}


# database endpoint example
@router.get("/list-index")
async def list_index():
    return s3vectors.list_indexes(vectorBucketName=config.vector_bucket)


@router.get("/list-vectors")
async def list_vectors():
    return s3vectors.list_vectors(
        vectorBucketName=config.vector_bucket, indexName=config.vector_index
    )
