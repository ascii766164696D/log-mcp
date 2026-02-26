#!/usr/bin/env python3
"""Generate realistic test log files for exercising log-mcp tools.

Produces test.log (~1500 lines, server1) and test2.log (~1500 lines, server2)
simulating two servers of the same web application with:
- Shared patterns at different frequencies (exercises frequency_outliers)
- Unique error patterns per server (exercises unique detection in compare_logs)
- Realistic prefixed identifiers (exercises <ID> normalization)
- Multi-line stack traces on errors
"""

from __future__ import annotations

import random
import string
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
random.seed(42)


def _rand_hex(n: int) -> str:
    return "".join(random.choices("0123456789abcdef", k=n))


def _rand_alnum(n: int) -> str:
    chars = string.ascii_lowercase + string.digits
    # Ensure at least one digit (for <ID> pattern matching)
    result = list(random.choices(chars, k=n))
    if not any(c.isdigit() for c in result):
        result[random.randint(0, n - 1)] = random.choice(string.digits)
    return "".join(result)


def _uuid() -> str:
    return f"{_rand_hex(8)}-{_rand_hex(4)}-{_rand_hex(4)}-{_rand_hex(4)}-{_rand_hex(12)}"


def _rand_ip(prefix: str) -> str:
    return f"{prefix}.{random.randint(1, 254)}.{random.randint(1, 254)}"


# ---------------------------------------------------------------------------
# Log entry generators
# ---------------------------------------------------------------------------

def _ts_str(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def health_check(ts: datetime, ip: str) -> list[str]:
    ms = random.randint(1, 8)
    return [
        f"{_ts_str(ts)} DEBUG Processing request GET /api/health from {ip}",
        f"{_ts_str(ts)} INFO Request completed: GET /api/health 200 {ms}ms",
    ]


def get_orders(ts: datetime, ip: str) -> list[str]:
    ms = random.randint(20, 4500)
    lines = [
        f"{_ts_str(ts)} DEBUG Processing request GET /api/orders from {ip}",
    ]
    if ms > 2000:
        uid = random.randint(10000, 99999)
        lines.append(
            f"{_ts_str(ts)} WARN Slow query detected: {ms}ms for SELECT * FROM orders WHERE user_id={uid}"
        )
    lines.append(f"{_ts_str(ts)} INFO Request completed: GET /api/orders 200 {ms}ms")
    return lines


def post_orders(ts: datetime, ip: str) -> list[str]:
    oid = random.randint(90000, 99999)
    uid = random.randint(10000, 99999)
    total = round(random.uniform(9.99, 999.99), 2)
    ms = random.randint(200, 1800)
    return [
        f"{_ts_str(ts)} DEBUG Processing request POST /api/orders from {ip}",
        f"{_ts_str(ts)} INFO Order created: order_id={oid} user_id={uid} total={total}",
        f"{_ts_str(ts)} INFO Request completed: POST /api/orders 201 {ms}ms",
    ]


def get_users(ts: datetime, ip: str) -> list[str]:
    uid = random.randint(10000, 99999)
    ms = random.randint(30, 300)
    return [
        f"{_ts_str(ts)} DEBUG Processing request GET /api/users/{uid} from {ip}",
        f"{_ts_str(ts)} INFO Request completed: GET /api/users/{uid} 200 {ms}ms",
    ]


def user_login(ts: datetime, _ip: str) -> list[str]:
    uid = random.randint(10000, 99999)
    sess = f"sess_{_rand_alnum(8)}"
    return [f"{_ts_str(ts)} INFO User login: user_id={uid} session={sess}"]


def user_logout(ts: datetime, _ip: str) -> list[str]:
    uid = random.randint(10000, 99999)
    dur = random.randint(30, 3600)
    return [f"{_ts_str(ts)} INFO User logout: user_id={uid} session_duration={dur}s"]


def get_inventory(ts: datetime, ip: str) -> list[str]:
    sku = f"SKU-{random.randint(1, 50):03d}"
    ms = random.randint(20, 150)
    return [
        f"{_ts_str(ts)} DEBUG Processing request GET /api/inventory/{sku} from {ip}",
        f"{_ts_str(ts)} INFO Request completed: GET /api/inventory/{sku} 200 {ms}ms",
    ]


def delete_order(ts: datetime, ip: str) -> list[str]:
    oid = random.randint(90000, 99999)
    ms = random.randint(100, 500)
    return [
        f"{_ts_str(ts)} DEBUG Processing request DELETE /api/orders/{oid} from {ip}",
        f"{_ts_str(ts)} INFO Order cancelled: order_id={oid}",
        f"{_ts_str(ts)} INFO Request completed: DELETE /api/orders/{oid} 200 {ms}ms",
    ]


def health_check_ok(ts: datetime, _ip: str) -> list[str]:
    return [f"{_ts_str(ts)} INFO Health check OK"]


def deprecated_api(ts: datetime, _ip: str) -> list[str]:
    version = random.choice(["v1", "v2"])
    client = random.choice([
        "app-mobile/2.3.1", "app-mobile/2.1.0", "app-web/1.5.2", "sdk-python/0.9.1",
    ])
    return [f"{_ts_str(ts)} WARN Deprecated API version {version} called by client {client}"]


def cache_hit_ratio(ts: datetime, _ip: str) -> list[str]:
    ratio = round(random.uniform(85.0, 99.0), 1)
    return [f"{_ts_str(ts)} INFO Cache hit ratio: {ratio}%"]


def scheduled_job(ts: datetime, _ip: str) -> list[str]:
    job = random.choice(["cleanup_expired_sessions", "sync_inventory", "flush_metrics", "rotate_logs"])
    count = random.randint(5, 500)
    ms = random.randint(500, 12000)
    return [
        f"{_ts_str(ts)} INFO Scheduled job started: {job}",
        f"{_ts_str(ts)} INFO Scheduled job completed: {job}, processed {count} items in {ms}ms",
    ]


def rate_limit_warn(ts: datetime, _ip: str) -> list[str]:
    current = random.randint(400, 980)
    limit = random.choice([500, 1000])
    return [f"{_ts_str(ts)} WARN Rate limit approaching: {current}/{limit} requests in current window"]


def inventory_low(ts: datetime, _ip: str) -> list[str]:
    sku = f"SKU-{random.randint(1, 50):03d}"
    units = random.randint(1, 4)
    return [f"{_ts_str(ts)} WARN Inventory low: {sku} has {units} units remaining (threshold: 5)"]


def conn_pool_exhausted(ts: datetime, _ip: str) -> list[str]:
    return [f"{_ts_str(ts)} WARN Connection pool exhausted, waiting for available connection"]


# ---------------------------------------------------------------------------
# Server1-only errors
# ---------------------------------------------------------------------------

def redis_cache_failure(ts: datetime, _ip: str) -> list[str]:
    host = random.choice(["redis-01.internal", "redis-02.internal"])
    handler = random.choice(["get_orders", "get_user", "get_inventory"])
    line = random.randint(80, 130)
    return [
        f"{_ts_str(ts)} ERROR Connection to cache server failed",
        f"redis.exceptions.ConnectionError: Connection refused to {host}:6379",
        f"    at app.cache.RedisPool.get_connection(pool.py:42)",
        f"    at app.api.handlers.{handler}(handlers.py:{line})",
        f"{_ts_str(ts)} WARN Falling back to direct DB query (cache miss)",
    ]


def gc_pause(ts: datetime, _ip: str) -> list[str]:
    ms = random.randint(50, 350)
    return [f"{_ts_str(ts)} DEBUG GC pause: {ms}ms"]


def slow_query_standalone(ts: datetime, _ip: str) -> list[str]:
    ms = random.randint(1500, 8000)
    table = random.choice(["orders", "users", "inventory", "sessions"])
    cond = random.choice(["status='pending'", "created_at > NOW() - INTERVAL '1 hour'", "active=true"])
    return [f"{_ts_str(ts)} WARN Slow query detected: {ms}ms for SELECT * FROM {table} WHERE {cond}"]


def memory_warning(ts: datetime, _ip: str) -> list[str]:
    pct = random.randint(82, 96)
    return [f"{_ts_str(ts)} WARN High memory usage: {pct}% (threshold: 80%)"]


def null_pointer_error(ts: datetime, _ip: str) -> list[str]:
    oid = random.randint(90000, 99999)
    attr = random.choice(["price", "quantity", "shipping_address"])
    return [
        f"{_ts_str(ts)} ERROR NullPointerError processing order {oid}",
        "Traceback (most recent call last):",
        '    File "app/api/handlers.py", line 235, in create_order',
        '    File "app/models/order.py", line 42, in validate',
        f"AttributeError: 'NoneType' object has no attribute '{attr}'",
    ]


def validation_error(ts: datetime, ip: str) -> list[str]:
    field = random.choice(["shipping_address", "payment_method", "email", "phone"])
    return [
        f"{_ts_str(ts)} ERROR Validation failed for order: missing required field '{field}'",
        f"{_ts_str(ts)} INFO Request completed: POST /api/orders 400 12ms",
    ]


def redis_reconnected(ts: datetime, _ip: str) -> list[str]:
    host = random.choice(["redis-01.internal", "redis-02.internal"])
    return [f"{_ts_str(ts)} INFO Cache server {host}:6379 reconnected"]


# ---------------------------------------------------------------------------
# Server2-only errors
# ---------------------------------------------------------------------------

def payment_timeout(ts: datetime, _ip: str) -> list[str]:
    req_id = f"req_{_rand_alnum(12)}"
    oid = random.randint(90000, 99999)
    attempts = random.choice([2, 3])
    lines = []
    for attempt in range(1, attempts + 1):
        lines.extend([
            f"{_ts_str(ts)} ERROR Timeout waiting for response from payment-service.internal:443 request={req_id}",
            f"TimeoutError: Request timed out after 30000ms",
            f"    at app.services.payment.charge(payment.py:78)",
            f"    at app.api.handlers.create_order(handlers.py:240)",
        ])
        if attempt < attempts:
            lines.append(f"{_ts_str(ts)} WARN Retrying payment request (attempt {attempt + 1}/{attempts})")
    lines.append(f"{_ts_str(ts)} ERROR Payment failed after {attempts} retries for order_id={oid}")
    lines.append(f"{_ts_str(ts)} INFO Request completed: POST /api/orders 503 {attempts * 30000 + 500}ms")
    return lines


def webhook_signature_error(ts: datetime, _ip: str) -> list[str]:
    evt = f"evt_{_rand_alnum(8)}"
    return [
        f"{_ts_str(ts)} ERROR Invalid webhook signature for event {evt}",
        "ValueError: HMAC signature mismatch",
        "    at app.webhooks.verify_signature(webhooks.py:25)",
        "    at app.api.handlers.handle_webhook(handlers.py:310)",
        f"{_ts_str(ts)} INFO Request completed: POST /api/webhooks 401 {random.randint(3, 15)}ms",
    ]


def rate_limit_exceeded(ts: datetime, ip: str) -> list[str]:
    req_id = f"req_{_rand_alnum(12)}"
    return [
        f"{_ts_str(ts)} ERROR Rate limit exceeded for {ip} request={req_id}",
        f"{_ts_str(ts)} INFO Request completed: GET /api/orders 429 2ms",
    ]


def cert_warning(ts: datetime, _ip: str) -> list[str]:
    days = random.randint(3, 30)
    service = random.choice([
        "payment-service.internal", "auth-service.internal", "notification-service.internal",
    ])
    return [f"{_ts_str(ts)} WARN Certificate for {service} expires in {days} days"]


def upstream_error(ts: datetime, _ip: str) -> list[str]:
    service = random.choice([
        "inventory-service.internal", "shipping-service.internal", "notification-service.internal",
    ])
    code = random.choice([502, 503, 504])
    pay_id = f"pay_{_rand_alnum(8)}"
    return [
        f"{_ts_str(ts)} ERROR Upstream service error: {service} returned {code} for transaction {pay_id}",
        f"    at app.services.client.request(client.py:55)",
        f"    at app.api.handlers.process_payment(handlers.py:280)",
    ]


# ---------------------------------------------------------------------------
# Startup block (shared)
# ---------------------------------------------------------------------------

def startup_block(ts: datetime, port: int, db_host: str, cache_hosts: list[str]) -> list[str]:
    return [
        f"{_ts_str(ts)} INFO Application started successfully",
        f"{_ts_str(ts + timedelta(seconds=1))} INFO Listening on port {port}",
        f"{_ts_str(ts + timedelta(seconds=2))} DEBUG Loading configuration from /etc/app/config.yaml",
        f"{_ts_str(ts + timedelta(seconds=3))} INFO Connected to database at {db_host}:5432",
        f"{_ts_str(ts + timedelta(seconds=4))} INFO Cache pool initialized: {', '.join(h + ':6379' for h in cache_hosts)}",
    ]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_log(
    target_lines: int,
    port: int,
    db_host: str,
    cache_hosts: list[str],
    ip_prefix: str,
    shared_weights: dict,
    unique_generators: list[tuple[float, callable]],
) -> list[str]:
    ts = datetime(2024, 1, 15, 8, 0, 0)
    lines = startup_block(ts, port, db_host, cache_hosts)
    ts += timedelta(seconds=10)

    # Build weighted generator list: (weight, generator)
    generators: list[tuple[float, callable]] = []
    for name, weight in shared_weights.items():
        generators.append((weight, globals()[name]))
    generators.extend(unique_generators)

    total_weight = sum(w for w, _ in generators)
    cumulative = []
    acc = 0.0
    for w, g in generators:
        acc += w / total_weight
        cumulative.append((acc, g))

    while len(lines) < target_lines:
        r = random.random()
        gen = cumulative[-1][1]
        for threshold, g in cumulative:
            if r <= threshold:
                gen = g
                break

        ip = _rand_ip(ip_prefix)
        new_lines = gen(ts, ip)
        lines.extend(new_lines)
        ts += timedelta(seconds=random.randint(2, 8))

    return lines


def main() -> None:
    # Shared pattern weights — different per server to create frequency outliers
    server1_shared = {
        "health_check": 8,
        "get_orders": 10,
        "post_orders": 12,  # server1 heavier on POST
        "get_users": 5,
        "user_login": 4,
        "user_logout": 3,
        "get_inventory": 4,
        "delete_order": 2,
        "health_check_ok": 8,
        "deprecated_api": 3,
        "cache_hit_ratio": 2,
        "scheduled_job": 2,
        "rate_limit_warn": 2,
        "inventory_low": 2,
        "conn_pool_exhausted": 2,
    }

    server2_shared = {
        "health_check": 18,  # server2 gets ~2x more health checks
        "get_orders": 20,    # server2 gets ~2x more GET /api/orders
        "post_orders": 5,    # server2 lighter on POST
        "get_users": 5,
        "user_login": 4,
        "user_logout": 3,
        "get_inventory": 4,
        "delete_order": 2,
        "health_check_ok": 18,
        "deprecated_api": 3,
        "cache_hit_ratio": 2,
        "scheduled_job": 2,
        "rate_limit_warn": 3,
        "inventory_low": 2,
        "conn_pool_exhausted": 1,
    }

    server1_unique = [
        (4, redis_cache_failure),
        (3, gc_pause),
        (3, slow_query_standalone),
        (3, memory_warning),
        (1.5, null_pointer_error),
        (1.5, validation_error),
        (1, redis_reconnected),
    ]

    server2_unique = [
        (3, payment_timeout),
        (4, webhook_signature_error),
        (2, rate_limit_exceeded),
        (2, cert_warning),
        (2, upstream_error),
    ]

    log1 = generate_log(
        target_lines=1500,
        port=8080,
        db_host="db-primary.internal",
        cache_hosts=["redis-01.internal", "redis-02.internal"],
        ip_prefix="192.168",
        shared_weights=server1_shared,
        unique_generators=server1_unique,
    )

    log2 = generate_log(
        target_lines=1500,
        port=9090,
        db_host="db-replica.internal",
        cache_hosts=["redis-03.internal", "redis-04.internal"],
        ip_prefix="10.0",
        shared_weights=server2_shared,
        unique_generators=server2_unique,
    )

    test1_path = REPO_ROOT / "test.log"
    test2_path = REPO_ROOT / "test2.log"

    test1_path.write_text("\n".join(log1) + "\n")
    test2_path.write_text("\n".join(log2) + "\n")

    print(f"Generated {test1_path} ({len(log1)} lines)")
    print(f"Generated {test2_path} ({len(log2)} lines)")


if __name__ == "__main__":
    main()
