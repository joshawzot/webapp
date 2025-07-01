from flask_caching import Cache
import redis

cache = Cache()
redis_client = redis.Redis(host='localhost', port=6379, db=0)