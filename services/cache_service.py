"""
Cache Service for ChatBI platform.
Provides Redis-based caching with intelligent cache management and performance optimization.
"""

import json
import pickle
import hashlib
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
import asyncio
import redis.asyncio as aioredis

from utils.config import get_redis_url
from utils.logger import get_logger
from utils.exceptions import ServiceUnavailableException, ErrorCodes

logger = get_logger(__name__)


class CacheService:
    """
    Comprehensive caching service with Redis backend, providing
    intelligent cache management, TTL handling, and performance optimization.
    """

    def __init__(self):
        """Initialize cache service with Redis connection."""
        self.redis_url = get_redis_url()
        self.redis_client = None
        self.connection_pool = None

        # Cache configuration
        self.default_ttl = 3600  # 1 hour
        self.max_key_length = 250
        self.key_prefix = "chatbi:"

        # Performance tracking
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }

        # Cache strategies
        self.cache_strategies = {
            "query_results": {"ttl": 1800, "compression": True},  # 30 minutes
            "table_metadata": {"ttl": 86400, "compression": False},  # 24 hours
            "user_permissions": {"ttl": 3600, "compression": False},  # 1 hour
            "chart_suggestions": {"ttl": 7200, "compression": True},  # 2 hours
            "session_data": {"ttl": 1800, "compression": False}  # 30 minutes
        }

        # Initialize connection
        asyncio.create_task(self._initialize_connection())

    async def _initialize_connection(self):
        """Initialize Redis connection with connection pooling."""
        try:
            self.connection_pool = aioredis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=20,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )

            self.redis_client = aioredis.Redis(
                connection_pool=self.connection_pool,
                decode_responses=False  # We'll handle encoding manually
            )

            # Test connection
            await self.redis_client.ping()
            logger.info("Cache service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize cache service: {e}")
            self.redis_client = None

    async def get(
            self,
            key: str,
            default: Any = None,
            strategy: Optional[str] = None
    ) -> Any:
        """
        Get value from cache.

        Args:
            key: Cache key
            default: Default value if key not found
            strategy: Cache strategy name for configuration

        Returns:
            Cached value or default
        """
        try:
            if not self.redis_client:
                logger.warning("Redis client not available")
                return default

            # Normalize key
            cache_key = self._normalize_key(key)

            # Get from Redis
            cached_data = await self.redis_client.get(cache_key)

            if cached_data is None:
                self.stats["misses"] += 1
                return default

            # Deserialize data
            value = self._deserialize(cached_data, strategy)
            self.stats["hits"] += 1

            logger.debug(f"Cache hit for key: {key}")
            return value

        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            self.stats["errors"] += 1
            return default

    async def set(
            self,
            key: str,
            value: Any,
            ttl: Optional[int] = None,
            strategy: Optional[str] = None
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            strategy: Cache strategy name for configuration

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.redis_client:
                logger.warning("Redis client not available")
                return False

            # Normalize key
            cache_key = self._normalize_key(key)

            # Get strategy configuration
            strategy_config = self.cache_strategies.get(strategy, {})
            ttl = ttl or strategy_config.get("ttl", self.default_ttl)

            # Serialize data
            serialized_data = self._serialize(value, strategy)

            # Set in Redis with TTL
            await self.redis_client.setex(cache_key, ttl, serialized_data)

            self.stats["sets"] += 1
            logger.debug(f"Cache set for key: {key}, TTL: {ttl}")
            return True

        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
            self.stats["errors"] += 1
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.redis_client:
                return False

            cache_key = self._normalize_key(key)
            result = await self.redis_client.delete(cache_key)

            if result > 0:
                self.stats["deletes"] += 1
                logger.debug(f"Cache delete for key: {key}")
                return True

            return False

        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {e}")
            self.stats["errors"] += 1
            return False

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key to check

        Returns:
            True if key exists, False otherwise
        """
        try:
            if not self.redis_client:
                return False

            cache_key = self._normalize_key(key)
            result = await self.redis_client.exists(cache_key)
            return result > 0

        except Exception as e:
            logger.error(f"Cache exists check failed for key {key}: {e}")
            return False

    async def get_ttl(self, key: str) -> int:
        """
        Get remaining TTL for a key.

        Args:
            key: Cache key

        Returns:
            Remaining TTL in seconds, -1 if key doesn't exist
        """
        try:
            if not self.redis_client:
                return -1

            cache_key = self._normalize_key(key)
            ttl = await self.redis_client.ttl(cache_key)
            return ttl

        except Exception as e:
            logger.error(f"Cache TTL check failed for key {key}: {e}")
            return -1

    async def extend_ttl(self, key: str, additional_seconds: int) -> bool:
        """
        Extend TTL for an existing key.

        Args:
            key: Cache key
            additional_seconds: Additional seconds to add to TTL

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.redis_client:
                return False

            cache_key = self._normalize_key(key)
            current_ttl = await self.redis_client.ttl(cache_key)

            if current_ttl > 0:
                new_ttl = current_ttl + additional_seconds
                result = await self.redis_client.expire(cache_key, new_ttl)
                return result

            return False

        except Exception as e:
            logger.error(f"Cache TTL extension failed for key {key}: {e}")
            return False

    async def get_multiple(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values from cache.

        Args:
            keys: List of cache keys

        Returns:
            Dictionary mapping keys to values
        """
        try:
            if not self.redis_client or not keys:
                return {}

            # Normalize keys
            cache_keys = [self._normalize_key(key) for key in keys]

            # Get values using pipeline for efficiency
            async with self.redis_client.pipeline() as pipe:
                for cache_key in cache_keys:
                    pipe.get(cache_key)

                results = await pipe.execute()

            # Process results
            result_dict = {}
            for i, (original_key, result) in enumerate(zip(keys, results)):
                if result is not None:
                    try:
                        result_dict[original_key] = self._deserialize(result)
                        self.stats["hits"] += 1
                    except Exception as e:
                        logger.warning(f"Failed to deserialize cached value for {original_key}: {e}")
                        self.stats["errors"] += 1
                else:
                    self.stats["misses"] += 1

            return result_dict

        except Exception as e:
            logger.error(f"Cache get_multiple failed: {e}")
            self.stats["errors"] += 1
            return {}

    async def set_multiple(
            self,
            data: Dict[str, Any],
            ttl: Optional[int] = None,
            strategy: Optional[str] = None
    ) -> int:
        """
        Set multiple values in cache.

        Args:
            data: Dictionary of key-value pairs
            ttl: Time to live in seconds
            strategy: Cache strategy name

        Returns:
            Number of successfully set keys
        """
        try:
            if not self.redis_client or not data:
                return 0

            # Get strategy configuration
            strategy_config = self.cache_strategies.get(strategy, {})
            ttl = ttl or strategy_config.get("ttl", self.default_ttl)

            successful_sets = 0

            # Use pipeline for efficiency
            async with self.redis_client.pipeline() as pipe:
                for key, value in data.items():
                    try:
                        cache_key = self._normalize_key(key)
                        serialized_data = self._serialize(value, strategy)
                        pipe.setex(cache_key, ttl, serialized_data)
                    except Exception as e:
                        logger.warning(f"Failed to prepare cache data for {key}: {e}")

                results = await pipe.execute()
                successful_sets = sum(1 for result in results if result)

            self.stats["sets"] += successful_sets
            return successful_sets

        except Exception as e:
            logger.error(f"Cache set_multiple failed: {e}")
            self.stats["errors"] += 1
            return 0

    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching a pattern.

        Args:
            pattern: Key pattern to match (supports wildcards)

        Returns:
            Number of keys deleted
        """
        try:
            if not self.redis_client:
                return 0

            # Normalize pattern
            cache_pattern = self._normalize_key(pattern)

            # Find matching keys
            keys = []
            async for key in self.redis_client.scan_iter(match=cache_pattern):
                keys.append(key)

            if keys:
                deleted_count = await self.redis_client.delete(*keys)
                self.stats["deletes"] += deleted_count
                logger.info(f"Invalidated {deleted_count} keys matching pattern: {pattern}")
                return deleted_count

            return 0

        except Exception as e:
            logger.error(f"Cache pattern invalidation failed for {pattern}: {e}")
            self.stats["errors"] += 1
            return 0

    async def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache information and statistics.

        Returns:
            Dictionary containing cache statistics and info
        """
        try:
            info = {
                "stats": self.stats.copy(),
                "connection_status": "connected" if self.redis_client else "disconnected",
                "strategies": self.cache_strategies.copy()
            }

            if self.redis_client:
                # Get Redis info
                redis_info = await self.redis_client.info()
                info["redis_info"] = {
                    "used_memory": redis_info.get("used_memory_human"),
                    "connected_clients": redis_info.get("connected_clients"),
                    "keyspace_hits": redis_info.get("keyspace_hits"),
                    "keyspace_misses": redis_info.get("keyspace_misses"),
                    "uptime_in_seconds": redis_info.get("uptime_in_seconds")
                }

                # Calculate hit rate
                total_requests = self.stats["hits"] + self.stats["misses"]
                info["hit_rate"] = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0

            return info

        except Exception as e:
            logger.error(f"Failed to get cache info: {e}")
            return {"error": str(e)}

    async def clear_all(self) -> bool:
        """
        Clear all cache data (use with caution).

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.redis_client:
                return False

            # Only clear keys with our prefix to avoid affecting other applications
            pattern = f"{self.key_prefix}*"
            deleted_count = 0

            async for key in self.redis_client.scan_iter(match=pattern):
                await self.redis_client.delete(key)
                deleted_count += 1

            logger.info(f"Cleared {deleted_count} cache entries")
            self.stats["deletes"] += deleted_count
            return True

        except Exception as e:
            logger.error(f"Cache clear all failed: {e}")
            self.stats["errors"] += 1
            return False

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on cache service.

        Returns:
            Health check results
        """
        try:
            if not self.redis_client:
                return {"status": "unhealthy", "error": "Redis client not available"}

            # Test basic operations
            test_key = f"{self.key_prefix}health_check"
            test_value = {"timestamp": datetime.now().isoformat(), "test": True}

            # Test set
            await self.redis_client.setex(test_key, 10, json.dumps(test_value))

            # Test get
            result = await self.redis_client.get(test_key)
            if result:
                retrieved_value = json.loads(result)

                # Test delete
                await self.redis_client.delete(test_key)

                return {
                    "status": "healthy",
                    "test_successful": True,
                    "response_time_ms": 0,  # Would measure actual response time
                    "connection_pool_size": len(
                        self.connection_pool._available_connections) if self.connection_pool else 0
                }
            else:
                return {"status": "unhealthy", "error": "Failed to retrieve test value"}

        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    # Private helper methods

    def _normalize_key(self, key: str) -> str:
        """Normalize cache key with prefix and length limits."""
        # Add prefix
        normalized_key = f"{self.key_prefix}{key}"

        # Truncate if too long and add hash to maintain uniqueness
        if len(normalized_key) > self.max_key_length:
            key_hash = hashlib.md5(normalized_key.encode()).hexdigest()[:8]
            truncated_key = normalized_key[:self.max_key_length - 9]  # Leave space for hash
            normalized_key = f"{truncated_key}_{key_hash}"

        return normalized_key

    def _serialize(self, value: Any, strategy: Optional[str] = None) -> bytes:
        """Serialize value for storage."""
        try:
            strategy_config = self.cache_strategies.get(strategy, {})
            use_compression = strategy_config.get("compression", False)

            if use_compression:
                # Use pickle for compression-friendly serialization
                serialized = pickle.dumps(value)
            else:
                # Use JSON for human-readable serialization
                serialized = json.dumps(value, default=str).encode('utf-8')

            return serialized

        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            # Fallback to string representation
            return str(value).encode('utf-8')

    def _deserialize(self, data: bytes, strategy: Optional[str] = None) -> Any:
        """Deserialize value from storage."""
        try:
            strategy_config = self.cache_strategies.get(strategy, {})
            use_compression = strategy_config.get("compression", False)

            if use_compression:
                # Try pickle first
                try:
                    return pickle.loads(data)
                except:
                    # Fallback to JSON
                    return json.loads(data.decode('utf-8'))
            else:
                # Try JSON first
                try:
                    return json.loads(data.decode('utf-8'))
                except:
                    # Fallback to string
                    return data.decode('utf-8')

        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            # Return raw data as string
            return data.decode('utf-8', errors='ignore')

    async def close(self):
        """Close Redis connection."""
        try:
            if self.redis_client:
                await self.redis_client.close()
            if self.connection_pool:
                await self.connection_pool.disconnect()
            logger.info("Cache service connection closed")
        except Exception as e:
            logger.error(f"Error closing cache service: {e}")


# Global cache service instance
cache_service = CacheService()


# Convenience functions for direct use
async def get_cached(key: str, default: Any = None, strategy: Optional[str] = None) -> Any:
    """Get value from cache."""
    return await cache_service.get(key, default, strategy)


async def set_cached(key: str, value: Any, ttl: Optional[int] = None, strategy: Optional[str] = None) -> bool:
    """Set value in cache."""
    return await cache_service.set(key, value, ttl, strategy)


async def delete_cached(key: str) -> bool:
    """Delete value from cache."""
    return await cache_service.delete(key)


async def invalidate_cache_pattern(pattern: str) -> int:
    """Invalidate cache keys matching pattern."""
    return await cache_service.invalidate_pattern(pattern)