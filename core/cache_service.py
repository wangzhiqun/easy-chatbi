import hashlib
import json
import pickle
from datetime import timedelta
from typing import Any, Optional, Union

import redis

from utils import logger, get_config


class CacheService:

    def __init__(self):
        self.config = get_config()
        self.redis_client = None
        self._connect()
        logger.info("Initialized Cache Service")

    def _connect(self):
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                decode_responses=False
            )
            self.redis_client.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            self.redis_client = None

    def _generate_key(self, key: str, namespace: str = "chatbi") -> str:
        return f"{namespace}:{key}"

    def _serialize(self, value: Any) -> bytes:
        try:
            if isinstance(value, (str, int, float, bool, list, dict)):
                return json.dumps(value).encode('utf-8')
            return pickle.dumps(value)
        except Exception as e:
            logger.error(f"Serialization failed: {str(e)}")
            raise

    def _deserialize(self, data: bytes) -> Any:
        try:
            try:
                return json.loads(data.decode('utf-8'))
            except:
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Deserialization failed: {str(e)}")
            raise

    def set(
            self,
            key: str,
            value: Any,
            ttl: Optional[Union[int, timedelta]] = None,
            namespace: str = "chatbi"
    ) -> bool:
        if not self.redis_client:
            return False

        try:
            cache_key = self._generate_key(key, namespace)
            serialized_value = self._serialize(value)

            if ttl:
                if isinstance(ttl, timedelta):
                    ttl = int(ttl.total_seconds())
                self.redis_client.setex(cache_key, ttl, serialized_value)
            else:
                self.redis_client.set(cache_key, serialized_value)

            logger.debug(f"Cached value for key: {cache_key}")
            return True

        except Exception as e:
            logger.error(f"Cache set failed: {str(e)}")
            return False

    def get(self, key: str, namespace: str = "chatbi") -> Optional[Any]:
        if not self.redis_client:
            return None

        try:
            cache_key = self._generate_key(key, namespace)
            data = self.redis_client.get(cache_key)

            if data is None:
                return None

            value = self._deserialize(data)
            logger.debug(f"Retrieved cached value for key: {cache_key}")
            return value

        except Exception as e:
            logger.error(f"Cache get failed: {str(e)}")
            return None

    def delete(self, key: str, namespace: str = "chatbi") -> bool:
        if not self.redis_client:
            return False

        try:
            cache_key = self._generate_key(key, namespace)
            result = self.redis_client.delete(cache_key)
            logger.debug(f"Deleted cache key: {cache_key}")
            return result > 0

        except Exception as e:
            logger.error(f"Cache delete failed: {str(e)}")
            return False

    def exists(self, key: str, namespace: str = "chatbi") -> bool:
        if not self.redis_client:
            return False

        try:
            cache_key = self._generate_key(key, namespace)
            return self.redis_client.exists(cache_key) > 0

        except Exception as e:
            logger.error(f"Cache exists check failed: {str(e)}")
            return False

    def clear_namespace(self, namespace: str = "chatbi") -> int:
        if not self.redis_client:
            return 0

        try:
            pattern = f"{namespace}:*"
            keys = self.redis_client.keys(pattern)

            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Cleared {deleted} keys from namespace: {namespace}")
                return deleted

            return 0

        except Exception as e:
            logger.error(f"Clear namespace failed: {str(e)}")
            return 0

    def get_ttl(self, key: str, namespace: str = "chatbi") -> Optional[int]:
        if not self.redis_client:
            return None

        try:
            cache_key = self._generate_key(key, namespace)
            ttl = self.redis_client.ttl(cache_key)

            if ttl == -2:
                return None
            elif ttl == -1:
                return -1

            return ttl

        except Exception as e:
            logger.error(f"Get TTL failed: {str(e)}")
            return None

    def cache_query_result(
            self,
            query: str,
            result: Any,
            ttl: int = 3600
    ) -> bool:

        query_hash = hashlib.md5(query.encode()).hexdigest()
        key = f"query:{query_hash}"

        cache_data = {
            'query': query,
            'result': result,
            'cached_at': __import__('datetime').datetime.now().isoformat()
        }

        return self.set(key, cache_data, ttl, namespace="queries")

    def get_cached_query(self, query: str) -> Optional[Any]:
        query_hash = hashlib.md5(query.encode()).hexdigest()
        key = f"query:{query_hash}"

        cache_data = self.get(key, namespace="queries")
        if cache_data:
            logger.info(f"Found cached result for query hash: {query_hash}")
            return cache_data.get('result')

        return None

    def cache_analysis_result(
            self,
            analysis_id: str,
            result: Any,
            ttl: int = 7200
    ) -> bool:
        key = f"analysis:{analysis_id}"
        return self.set(key, result, ttl, namespace="analysis")

    def get_cached_analysis(self, analysis_id: str) -> Optional[Any]:
        key = f"analysis:{analysis_id}"
        return self.get(key, namespace="analysis")

    def cache_chart_config(
            self,
            chart_id: str,
            config: dict,
            ttl: int = 86400
    ) -> bool:
        key = f"chart:{chart_id}"
        return self.set(key, config, ttl, namespace="charts")

    def get_cached_chart(self, chart_id: str) -> Optional[dict]:
        key = f"chart:{chart_id}"
        return self.get(key, namespace="charts")

    def get_cache_stats(self) -> dict:
        if not self.redis_client:
            return {'connected': False}

        try:
            info = self.redis_client.info()
            stats = {
                'connected': True,
                'used_memory': info.get('used_memory_human', 'N/A'),
                'connected_clients': info.get('connected_clients', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_rate': 0
            }

            hits = stats['keyspace_hits']
            misses = stats['keyspace_misses']
            if hits + misses > 0:
                stats['hit_rate'] = round(hits / (hits + misses) * 100, 2)

            return stats

        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {'connected': False, 'error': str(e)}
