"""
API Cache Utility for External Vendor APIs

Provides in-memory caching with TTL (Time To Live) for external API calls.
Reduces API rate limit issues and improves response times.

Usage:
    from utils.api_cache import APICache
    
    cache = APICache(default_ttl=10)  # 10 second default TTL
    
    # Cache a function call
    @cache.cached(ttl=10)
    async def fetch_stock_price(symbol: str):
        # API call here
        return price
    
    # Or use directly
    cache_key = cache.generate_key(url, params, api_key)
    cached_data = cache.get(cache_key)
    if cached_data is None:
        data = await make_api_call()
        cache.set(cache_key, data, ttl=10)
"""

import hashlib
import time
import threading
from typing import Any, Dict, Optional, Callable, Tuple
from functools import wraps
from urllib.parse import urlparse, parse_qs, urlencode
import logging

logger = logging.getLogger(__name__)


class APICache:
    """
    In-memory API cache with TTL support.
    Thread-safe for async operations.
    """
    
    def __init__(self, default_ttl: int = 10, cleanup_interval: int = 60):
        """
        Initialize API cache.
        
        Args:
            default_ttl: Default time-to-live in seconds (default: 10)
            cleanup_interval: How often to clean expired entries in seconds (default: 60)
        """
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self._cache: Dict[str, Tuple[Any, float]] = {}  # key -> (data, expiration_time)
        self._lock = threading.RLock()
        self._last_cleanup = time.time()
        
    def generate_key(self, url: str, params: Optional[Dict[str, Any]] = None, 
                     api_key: Optional[str] = None) -> str:
        """
        Generate a cache key from URL, parameters, and API key.
        
        Args:
            url: API endpoint URL
            params: Query parameters (will be sorted for consistency)
            api_key: API key (will be hashed for security)
            
        Returns:
            Cache key string
        """
        # Normalize URL (remove fragment, sort query params)
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        
        # Merge URL query params with provided params
        url_params = parse_qs(parsed.query)
        if params:
            for key, value in params.items():
                url_params[key] = [str(value)]
        
        # Sort parameters for consistent keys
        sorted_params = sorted(url_params.items())
        params_str = urlencode(sorted_params, doseq=True)
        
        # Create key components
        key_parts = [base_url, params_str]
        
        # Hash API key if provided (don't store it in plain text)
        if api_key:
            api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
            key_parts.append(api_key_hash)
        
        # Generate final key
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get cached data if it exists and hasn't expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if not found/expired
        """
        with self._lock:
            # Periodic cleanup
            self._cleanup_if_needed()
            
            if key not in self._cache:
                return None
            
            data, expiration = self._cache[key]
            
            # Check if expired
            if time.time() > expiration:
                del self._cache[key]
                return None
            
            logger.debug(f"Cache HIT for key: {key[:16]}...")
            return data
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Store data in cache with TTL.
        
        Args:
            key: Cache key
            value: Data to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        with self._lock:
            ttl = ttl or self.default_ttl
            expiration = time.time() + ttl
            self._cache[key] = (value, expiration)
            logger.debug(f"Cache SET for key: {key[:16]}... (TTL: {ttl}s)")
    
    def delete(self, key: str) -> None:
        """Delete a cache entry."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Cache DELETE for key: {key[:16]}...")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")
    
    def _cleanup_if_needed(self) -> None:
        """Remove expired entries if cleanup interval has passed."""
        current_time = time.time()
        if current_time - self._last_cleanup < self.cleanup_interval:
            return
        
        self._last_cleanup = current_time
        expired_keys = [
            key for key, (_, expiration) in self._cache.items()
            if current_time > expiration
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def cached(self, ttl: Optional[int] = None, key_func: Optional[Callable] = None):
        """
        Decorator to cache function results.
        
        Args:
            ttl: Time-to-live in seconds (uses default if None)
            key_func: Optional function to generate cache key from function args
            
        Usage:
            @cache.cached(ttl=10)
            async def fetch_data(symbol: str):
                return await api_call(symbol)
        """
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Default: use function name + args + kwargs
                    key_parts = [func.__name__]
                    key_parts.extend(str(arg) for arg in args)
                    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                    cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
                
                # Check cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Call function and cache result
                result = await func(*args, **kwargs)
                self.set(cache_key, result, ttl=ttl)
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Default: use function name + args + kwargs
                    key_parts = [func.__name__]
                    key_parts.extend(str(arg) for arg in args)
                    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                    cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
                
                # Check cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Call function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl=ttl)
                return result
            
            # Return appropriate wrapper based on function type
            import inspect
            if inspect.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            current_time = time.time()
            total_entries = len(self._cache)
            expired_count = sum(
                1 for _, expiration in self._cache.values()
                if current_time > expiration
            )
            
            return {
                "total_entries": total_entries,
                "active_entries": total_entries - expired_count,
                "expired_entries": expired_count,
                "default_ttl": self.default_ttl,
            }


# Global cache instance (can be imported and used directly)
_global_cache: Optional[APICache] = None


def get_cache(default_ttl: int = 10) -> APICache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = APICache(default_ttl=default_ttl)
    return _global_cache
