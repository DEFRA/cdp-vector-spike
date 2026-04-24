import json
from logging import getLogger

import boto3

from app.config import config

bedrock = boto3.client("bedrock-runtime", region_name="eu-west-2")
s3vectors = boto3.client("s3vectors", region_name="eu-west-2")
logger = getLogger(__name__)


def ensure_index():
    try:
        response = s3vectors.get_index(
            vectorBucketName=config.vector_bucket, indexName=config.vector_index
        )
        logger.info("index exists %s", response["index"]["indexArn"])

    except s3vectors.exceptions.NotFoundException:
        logger.info("creating vector index %s", config.vector_index)

        response = s3vectors.create_index(
            vectorBucketName=config.vector_bucket,
            indexName=config.vector_index,
            dimension=3,
            distanceMetric="cosine",
            dataType="float32",
            metadataConfiguration={
                "nonFilterableMetadataKeys": ["source_text", "filename"]
            },
        )


def store_embeddings(filename: str, text: str):
    logger.info("generating vector for %s", filename)
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0", body=json.dumps({"inputText": text})
    )

    # Extract embedding from response.
    response_body = json.loads(response["body"].read())
    vector = {
        "key": filename,
        "data": {"float32": response_body["embedding"]},
        "metadata": {"source_text": text, "filename": filename},
    }
    logger.info("storing vector for %s", filename)
    s3vectors.put_vectors(
        vectorBucketName=config.vector_bucket,
        indexName=config.vector_index,
        vectors=[vector],
    )
    logger.info("done storing vector for %s", filename)
