from typing import List, Dict, Any, Optional

from langchain_openai import OpenAIEmbeddings
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)

from utils import logger, get_config


class VectorService:

    def __init__(self):
        self.config = get_config()
        self.embeddings = OpenAIEmbeddings(api_key=self.config.openai_api_key)
        self.collection_name = "chatbi_vectors"
        self.collection = None
        self._connect()
        self._init_collection()
        logger.info("Initialized Vector Service")

    def _connect(self):
        try:
            connections.connect(
                alias="default",
                host=self.config.milvus_host,
                port=self.config.milvus_port
            )
            logger.info("Connected to Milvus vector database")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")

    def _init_collection(self):
        try:
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            else:
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="metadata", dtype=DataType.JSON),
                    FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=50)
                ]

                schema = CollectionSchema(
                    fields=fields,
                    description="ChatBI vector collection for semantic search"
                )

                self.collection = Collection(
                    name=self.collection_name,
                    schema=schema
                )

                index_params = {
                    "index_type": "IVF_FLAT",
                    "metric_type": "L2",
                    "params": {"nlist": 128}
                }
                self.collection.create_index(
                    field_name="embedding",
                    index_params=index_params
                )

                logger.info(f"Created new collection: {self.collection_name}")

            self.collection.load()

        except Exception as e:
            logger.error(f"Failed to initialize collection: {str(e)}")

    def embed_text(self, text: str) -> List[float]:
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            return []

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.embeddings.embed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            return []

    def insert_vectors(
            self,
            texts: List[str],
            metadatas: Optional[List[Dict[str, Any]]] = None,
            doc_type: str = "document"
    ) -> List[int]:
        try:
            if not self.collection:
                logger.error("Collection not initialized")
                return []

            embeddings = self.embed_texts(texts)

            if not embeddings:
                return []

            data = {
                "embedding": embeddings,
                "text": texts,
                "metadata": metadatas or [{}] * len(texts),
                "type": [doc_type] * len(texts)
            }

            result = self.collection.insert(data)

            self.collection.flush()

            logger.info(f"Inserted {len(texts)} vectors into collection")
            return result.primary_keys

        except Exception as e:
            logger.error(f"Failed to insert vectors: {str(e)}")
            return []

    def search_similar(
            self,
            query: str,
            top_k: int = 5,
            doc_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        try:
            if not self.collection:
                logger.error("Collection not initialized")
                return []

            query_embedding = self.embed_text(query)

            if not query_embedding:
                return []

            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }

            expr = f'type == "{doc_type}"' if doc_type else None

            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["text", "metadata", "type"]
            )

            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        'id': hit.id,
                        'distance': hit.distance,
                        'score': 1 / (1 + hit.distance),
                        'text': hit.entity.get('text'),
                        'metadata': hit.entity.get('metadata'),
                        'type': hit.entity.get('type')
                    })

            logger.info(f"Found {len(formatted_results)} similar documents")
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def delete_vectors(self, ids: List[int]) -> bool:
        try:
            if not self.collection:
                logger.error("Collection not initialized")
                return False

            expr = f"id in {ids}"
            self.collection.delete(expr)

            logger.info(f"Deleted {len(ids)} vectors")
            return True

        except Exception as e:
            logger.error(f"Failed to delete vectors: {str(e)}")
            return False

    def store_query_history(self, query: str, result: str, metadata: Optional[Dict] = None):
        try:
            text = f"Query: {query}\nResult: {result[:500]}"

            metadata = metadata or {}
            metadata.update({
                'query': query,
                'timestamp': __import__('datetime').datetime.now().isoformat()
            })

            self.insert_vectors(
                texts=[text],
                metadatas=[metadata],
                doc_type="query_history"
            )

            logger.info("Stored query in vector history")

        except Exception as e:
            logger.error(f"Failed to store query history: {str(e)}")

    def find_similar_queries(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        return self.search_similar(query, top_k, doc_type="query_history")

    def store_knowledge(self, title: str, content: str, metadata: Optional[Dict] = None):
        try:
            text = f"Title: {title}\n{content}"

            metadata = metadata or {}
            metadata['title'] = title

            self.insert_vectors(
                texts=[text],
                metadatas=[metadata],
                doc_type="knowledge"
            )

            logger.info(f"Stored knowledge document: {title}")

        except Exception as e:
            logger.error(f"Failed to store knowledge: {str(e)}")

    def search_knowledge(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.search_similar(query, top_k, doc_type="knowledge")

    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            if not self.collection:
                return {'error': 'Collection not initialized'}

            stats = {
                'collection_name': self.collection_name,
                'num_entities': self.collection.num_entities,
                'loaded': utility.load_state(self.collection_name),
                'schema': str(self.collection.schema)
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {'error': str(e)}

    def clear_collection(self) -> bool:
        try:
            if not self.collection:
                logger.error("Collection not initialized")
                return False

            self.collection.drop()
            self._init_collection()

            logger.info("Cleared collection")
            return True

        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            return False
