/**
 * API Cache Utility for External Vendor APIs
 * 
 * Provides in-memory caching with TTL (Time To Live) for external API calls.
 * Reduces API rate limit issues and improves response times.
 * 
 * Usage:
 *   import { APICache, cachedRequest } from './utils/apiCache';
 *   
 *   const cache = new APICache(10); // 10 second default TTL
 *   
 *   // Use cached request wrapper
 *   const data = await cachedRequest(
 *     axios.get('https://api.example.com/data', { params }),
 *     cache,
 *     'https://api.example.com/data',
 *     params,
 *     apiKey
 *   );
 */

import axios, { AxiosRequestConfig, AxiosResponse } from 'axios';
import crypto from 'crypto';
import { URL } from 'url';

interface CacheEntry<T> {
  data: T;
  expiration: number;
}

export class APICache {
  private cache: Map<string, CacheEntry<any>>;
  private defaultTTL: number;
  private cleanupInterval: number;
  private lastCleanup: number;

  constructor(defaultTTL: number = 10, cleanupInterval: number = 60) {
    this.defaultTTL = defaultTTL;
    this.cleanupInterval = cleanupInterval;
    this.cache = new Map();
    this.lastCleanup = Date.now();
  }

  /**
   * Generate a cache key from URL, parameters, and API key.
   */
  generateKey(url: string, params?: Record<string, any>, apiKey?: string): string {
    try {
      // Normalize URL
      const urlObj = new URL(url);
      const baseUrl = `${urlObj.protocol}//${urlObj.host}${urlObj.pathname}`;
      
      // Merge URL query params with provided params
      const allParams: Record<string, string> = {};
      
      // Add URL query params
      urlObj.searchParams.forEach((value, key) => {
        allParams[key] = value;
      });
      
      // Add provided params
      if (params) {
        Object.keys(params).forEach(key => {
          allParams[key] = String(params[key]);
        });
      }
      
      // Sort parameters for consistent keys
      const sortedParams = Object.keys(allParams)
        .sort()
        .map(key => `${key}=${allParams[key]}`)
        .join('&');
      
      // Create key components
      const keyParts = [baseUrl, sortedParams];
      
      // Hash API key if provided
      if (apiKey) {
        const apiKeyHash = crypto.createHash('sha256').update(apiKey).digest('hex').substring(0, 16);
        keyParts.push(apiKeyHash);
      }
      
      // Generate final key
      const keyString = keyParts.join('|');
      return crypto.createHash('md5').update(keyString).digest('hex');
    } catch (error) {
      // Fallback if URL parsing fails
      const keyString = `${url}|${JSON.stringify(params || {})}|${apiKey || ''}`;
      return crypto.createHash('md5').update(keyString).digest('hex');
    }
  }

  /**
   * Get cached data if it exists and hasn't expired.
   */
  get<T>(key: string): T | null {
    // Periodic cleanup
    this.cleanupIfNeeded();
    
    const entry = this.cache.get(key);
    if (!entry) {
      return null;
    }
    
    // Check if expired
    if (Date.now() > entry.expiration) {
      this.cache.delete(key);
      return null;
    }
    
    console.debug(`Cache HIT for key: ${key.substring(0, 16)}...`);
    return entry.data as T;
  }

  /**
   * Store data in cache with TTL.
   */
  set<T>(key: string, value: T, ttl?: number): void {
    const ttlToUse = ttl || this.defaultTTL;
    const expiration = Date.now() + (ttlToUse * 1000);
    this.cache.set(key, { data: value, expiration });
    console.debug(`Cache SET for key: ${key.substring(0, 16)}... (TTL: ${ttlToUse}s)`);
  }

  /**
   * Delete a cache entry.
   */
  delete(key: string): void {
    if (this.cache.has(key)) {
      this.cache.delete(key);
      console.debug(`Cache DELETE for key: ${key.substring(0, 16)}...`);
    }
  }

  /**
   * Clear all cache entries.
   */
  clear(): void {
    this.cache.clear();
    console.info('Cache cleared');
  }

  /**
   * Remove expired entries if cleanup interval has passed.
   */
  private cleanupIfNeeded(): void {
    const currentTime = Date.now();
    if (currentTime - this.lastCleanup < this.cleanupInterval * 1000) {
      return;
    }
    
    this.lastCleanup = currentTime;
    const expiredKeys: string[] = [];
    
    this.cache.forEach((entry, key) => {
      if (currentTime > entry.expiration) {
        expiredKeys.push(key);
      }
    });
    
    expiredKeys.forEach(key => this.cache.delete(key));
    
    if (expiredKeys.length > 0) {
      console.debug(`Cleaned up ${expiredKeys.length} expired cache entries`);
    }
  }

  /**
   * Get cache statistics.
   */
  stats(): { totalEntries: number; activeEntries: number; expiredEntries: number; defaultTTL: number } {
    const currentTime = Date.now();
    let activeCount = 0;
    let expiredCount = 0;
    
    this.cache.forEach(entry => {
      if (currentTime > entry.expiration) {
        expiredCount++;
      } else {
        activeCount++;
      }
    });
    
    return {
      totalEntries: this.cache.size,
      activeEntries: activeCount,
      expiredEntries: expiredCount,
      defaultTTL: this.defaultTTL
    };
  }
}

/**
 * Wrapper function to make cached axios requests.
 * 
 * @param requestFn Function that returns an axios promise
 * @param cache APICache instance
 * @param url Request URL
 * @param params Request parameters
 * @param apiKey Optional API key
 * @param ttl Optional TTL override
 */
export async function cachedRequest<T>(
  requestFn: () => Promise<AxiosResponse<T>>,
  cache: APICache,
  url: string,
  params?: Record<string, any>,
  apiKey?: string,
  ttl?: number
): Promise<T> {
  // Generate cache key
  const cacheKey = cache.generateKey(url, params, apiKey);
  
  // Check cache first
  const cachedData = cache.get<T>(cacheKey);
  if (cachedData !== null) {
    return cachedData;
  }
  
  // Make API call
  const response = await requestFn();
  const data = response.data;
  
  // Cache the result
  cache.set(cacheKey, data, ttl);
  
  return data;
}

/**
 * Global cache instance (can be imported and used directly)
 */
let globalCache: APICache | null = null;

export function getCache(defaultTTL: number = 10): APICache {
  if (!globalCache) {
    globalCache = new APICache(defaultTTL);
  }
  return globalCache;
}
