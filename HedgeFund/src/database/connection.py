"""
Module: Database Connection Manager
Purpose: Manage PostgreSQL and Redis connections with pooling
Dependencies: psycopg2, redis, sqlalchemy
Exports: get_db_connection, get_redis_connection, DatabaseManager

Provides connection pooling and health checks for all database operations.
"""

import psycopg2
from psycopg2 import pool
import redis
from typing import Optional
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages database connections with pooling
    
    Handles both PostgreSQL and Redis connections.
    Implements connection pooling for efficiency.
    """
    
    def __init__(self, config):
        """
        Initialize database manager
        
        Args:
            config: DatabaseConfig from configuration manager
        """
        self.config = config
        self.pg_pool: Optional[pool.SimpleConnectionPool] = None
        self.redis_client: Optional[redis.Redis] = None
        
        # Initialize connections
        self._init_postgres()
        self._init_redis()
    
    def _init_postgres(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.pg_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=self.config.pool_size,
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_database,
                user=self.config.postgres_user,
                password=self.config.postgres_password
            )
            
            logger.info(f"PostgreSQL pool initialized: {self.config.postgres_host}:{self.config.postgres_port}/{self.config.postgres_database}")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            raise
    
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=True,  # Auto-decode bytes to strings
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            
            logger.info(f"Redis connected: {self.config.redis_host}:{self.config.redis_port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """
        Get PostgreSQL connection from pool (context manager)
        
        Usage:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM trades")
        """
        conn = None
        try:
            conn = self.pg_pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                self.pg_pool.putconn(conn)
    
    def execute(self, query: str, params: tuple = None):
        """
        Execute a query (INSERT, UPDATE, DELETE)
        
        Args:
            query: SQL query
            params: Query parameters (for parameterized queries)
        
        Returns:
            Number of affected rows
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            affected = cursor.rowcount
            cursor.close()
            return affected
    
    def query(self, query: str, params: tuple = None) -> list:
        """
        Execute a SELECT query
        
        Args:
            query: SQL query
            params: Query parameters
        
        Returns:
            List of rows (as dictionaries)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()
            return results
    
    def query_one(self, query: str, params: tuple = None) -> Optional[dict]:
        """
        Execute a SELECT query expecting single result
        
        Returns:
            Single row as dictionary, or None
        """
        results = self.query(query, params)
        return results[0] if results else None
    
    def get_redis(self) -> redis.Redis:
        """Get Redis client"""
        return self.redis_client
    
    def health_check(self) -> dict:
        """
        Check health of all database connections
        
        Returns:
            Dict with health status
        """
        health = {
            'postgres': False,
            'redis': False
        }
        
        # Check PostgreSQL
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
            health['postgres'] = True
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
        
        # Check Redis
        try:
            self.redis_client.ping()
            health['redis'] = True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
        
        return health
    
    def close(self):
        """Close all connections"""
        if self.pg_pool:
            self.pg_pool.closeall()
            logger.info("PostgreSQL pool closed")
        
        if self.redis_client:
            self.redis_client.close()
            logger.info("Redis connection closed")


# Singleton instance (initialized once)
_db_manager: Optional[DatabaseManager] = None


def initialize_database(config):
    """Initialize global database manager"""
    global _db_manager
    _db_manager = DatabaseManager(config)
    return _db_manager


def get_database() -> DatabaseManager:
    """Get database manager instance"""
    if _db_manager is None:
        raise RuntimeError("Database not initialized. Call initialize_database() first.")
    return _db_manager


if __name__ == "__main__":
    """Test database connections"""
    import sys
    sys.path.insert(0, '/mnt/4tb/stock_scanner/market_news/HedgeFund')
    
    from src.common.config import load_config
    
    print("Testing Database Connections...")
    
    # Load config
    config = load_config()
    
    print(f"\n📊 Configuration:")
    print(f"  PostgreSQL: {config.database.postgres_host}:{config.database.postgres_port}")
    print(f"  Database: {config.database.postgres_database}")
    print(f"  Redis: {config.database.redis_host}:{config.database.redis_port}")
    
    # Try to initialize (will fail if postgres/redis not running - that's OK for now)
    print(f"\n🔌 Testing connections...")
    
    try:
        db = DatabaseManager(config.database)
        
        # Health check
        health = db.health_check()
        print(f"\n✓ Health check results:")
        print(f"  PostgreSQL: {'✅' if health['postgres'] else '❌'}")
        print(f"  Redis: {'✅' if health['redis'] else '❌'}")
        
        if health['postgres']:
            # Test query
            result = db.query_one("SELECT NOW() as current_time")
            print(f"  Current DB time: {result['current_time']}")
        
        if health['redis']:
            # Test Redis
            db.get_redis().set('test_key', 'test_value')
            value = db.get_redis().get('test_key')
            print(f"  Redis test: {value}")
            db.get_redis().delete('test_key')
        
        db.close()
        
        print("\n✅ Database connections working!")
        
    except Exception as e:
        print(f"\n⚠️  Database connection test: {e}")
        print("   This is expected if PostgreSQL/Redis not running yet.")
        print("   To set up databases:")
        print("   1. Install PostgreSQL: apt install postgresql")
        print("   2. Install Redis: apt install redis-server")
        print("   3. Run schema: psql -U postgres < src/database/schema.sql")


