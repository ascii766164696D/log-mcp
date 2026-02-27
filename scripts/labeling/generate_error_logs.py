"""Generate synthetic error-heavy log files for training data augmentation.

Creates 6 realistic log files (~500 lines each, ~70-80% LOOK-worthy) covering
error patterns underrepresented in the loghub dataset:

  - JavaApp:        Stack traces, NPE, OOM, ClassNotFound, Spring Boot errors
  - KernelErrors:   Kernel panics, OOM kills, segfaults, disk I/O, MCE events
  - DatabaseErrors: PG/MySQL deadlocks, replication lag, slow queries, disk full
  - WebServer:      nginx/Apache 5xx, upstream timeouts, SSL failures
  - Microservices:  gRPC deadline exceeded, circuit breaker, CrashLoopBackOff
  - SecurityEvents: SSH brute force, failed auth, cert expiry, firewall drops

Usage:
    uv run python -m scripts.labeling.generate_error_logs
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

from scripts.labeling.config import LOGHUB_DIR

random.seed(42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(base: datetime, offset_s: int) -> str:
    """ISO-style timestamp."""
    t = base + timedelta(seconds=offset_s)
    return t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _ts_kernel(base: datetime, offset_s: int) -> str:
    """Kernel-style [seconds.usec] timestamp."""
    return f"[{offset_s:>10.6f}]"


def _ts_syslog(base: datetime, offset_s: int) -> str:
    """Syslog-style timestamp."""
    t = base + timedelta(seconds=offset_s)
    return t.strftime("%b %d %H:%M:%S")


# ---------------------------------------------------------------------------
# JavaApp
# ---------------------------------------------------------------------------

_JAVA_CLASSES = [
    "com.myapp.service.UserService",
    "com.myapp.service.OrderService",
    "com.myapp.controller.ApiController",
    "com.myapp.repository.UserRepository",
    "com.myapp.config.DataSourceConfig",
    "com.myapp.messaging.KafkaConsumer",
    "com.myapp.cache.RedisCacheManager",
    "org.springframework.beans.factory.BeanFactory",
]

_JAVA_STACK_METHODS = [
    "processRequest", "handleOrder", "findById", "getConnection",
    "initializeBean", "consumeMessage", "evictCache", "authenticate",
]

def _java_stack_trace(exc_class: str, message: str, depth: int = 5) -> list[str]:
    """Generate a realistic Java stack trace."""
    lines = [f"{exc_class}: {message}"]
    for i in range(depth):
        cls = random.choice(_JAVA_CLASSES)
        method = random.choice(_JAVA_STACK_METHODS)
        lineno = random.randint(30, 500)
        lines.append(f"\tat {cls}.{method}({cls.split('.')[-1]}.java:{lineno})")
    lines.append(f"\tat java.base/java.lang.Thread.run(Thread.java:{random.randint(800,900)})")
    return lines


def generate_java_app(base: datetime) -> list[str]:
    lines: list[str] = []
    t = 0

    def log(level: str, cls: str, msg: str):
        nonlocal t
        t += random.randint(0, 3)
        lines.append(f"{_ts(base, t)} [{level:>5s}] {cls} - {msg}")

    def stack(exc: str, msg: str, level: str = "ERROR", cls: str = ""):
        if not cls:
            cls = random.choice(_JAVA_CLASSES)
        log(level, cls, f"{exc}: {msg}")
        for sl in _java_stack_trace(exc, msg)[1:]:
            lines.append(sl)

    # Normal startup
    log("INFO", "o.s.b.SpringApplication", "Starting MyApp v2.4.1 using Java 17.0.2")
    log("INFO", "o.s.b.SpringApplication", "Active profiles: production")
    log("INFO", "o.s.d.r.c.RepositoryConfigurationDelegate", "Bootstrapping Spring Data repositories")
    log("INFO", "o.s.b.w.e.t.TomcatWebServer", "Tomcat initialized with port(s): 8080 (http)")
    log("INFO", "o.a.c.c.C.[Tomcat].[localhost].[/]", "Initializing Spring embedded WebApplicationContext")
    log("INFO", "c.m.config.DataSourceConfig", "Initializing HikariCP connection pool")
    log("INFO", "c.z.h.HikariDataSource", "HikariPool-1 - Start completed.")
    log("INFO", "o.s.b.SpringApplication", "Started MyApp in 8.234 seconds")

    for _ in range(120):
        r = random.random()
        if r < 0.12:
            stack("java.lang.NullPointerException", random.choice([
                "Cannot invoke method on null reference",
                "null",
                "Cannot read field \"id\" because \"user\" is null",
                "Cannot invoke \"String.length()\" because \"str\" is null",
            ]))
        elif r < 0.22:
            stack("java.lang.OutOfMemoryError", random.choice([
                "Java heap space",
                "GC overhead limit exceeded",
                "Metaspace",
                "Direct buffer memory",
            ]), "FATAL")
        elif r < 0.30:
            stack("java.lang.ClassNotFoundException", random.choice([
                "com.myapp.legacy.OldService",
                "com.myapp.plugins.ExtPlugin",
                "org.apache.kafka.common.serialization.CustomDeserializer",
            ]))
        elif r < 0.38:
            stack("org.springframework.beans.factory.BeanCreationException",
                  f"Error creating bean with name '{random.choice(['userService', 'orderService', 'cacheManager'])}': Injection of autowired dependencies failed")
        elif r < 0.46:
            log("ERROR", "c.z.h.p.HikariPool", f"HikariPool-1 - Connection is not available, request timed out after {random.randint(30000,60000)}ms.")
            log("WARN", "c.z.h.p.HikariPool", f"HikariPool-1 - Apparent connection leak detected (open for {random.randint(60,300)} seconds)")
        elif r < 0.52:
            stack("java.sql.SQLException", random.choice([
                "Connection refused to host: db-primary.internal:5432",
                "Too many connections",
                f"Statement cancelled due to timeout after {random.randint(30,120)} seconds",
                "Deadlock found when trying to get lock; try restarting transaction",
            ]))
        elif r < 0.58:
            stack("org.apache.kafka.common.errors.TimeoutException",
                  f"Topic partition-{random.randint(0,11)} not present in metadata after {random.randint(60000,120000)}ms")
        elif r < 0.64:
            log("WARN", "c.m.service.OrderService", f"Retry attempt {random.randint(2,5)}/5 for order #{random.randint(10000,99999)} after transient failure")
            log("WARN", "c.m.service.OrderService", f"Circuit breaker 'orderService' is now OPEN after {random.randint(5,20)} consecutive failures")
        elif r < 0.70:
            log("ERROR", "c.m.messaging.KafkaConsumer", f"Failed to deserialize message from topic orders-{random.choice(['created','updated'])}")
            stack("com.fasterxml.jackson.databind.JsonMappingException",
                  f"Unexpected token (START_ARRAY), expected VALUE_STRING at [line: 1, column: {random.randint(10,200)}]")
        elif r < 0.75:
            log("WARN", "o.s.s.w.a.ExceptionTranslationFilter", f"Authentication failed: Bad credentials for user '{random.choice(['admin', 'service-account', 'deploy-bot'])}'")
        elif r < 0.80:
            log("ERROR", "c.m.cache.RedisCacheManager", f"Redis connection failed: Connection refused: redis-{random.randint(1,3)}.internal:6379")
        elif r < 0.85:
            log("INFO", "c.m.controller.ApiController", f"GET /api/v1/users/{random.randint(1,9999)} - 200 OK ({random.randint(5,50)}ms)")
        elif r < 0.90:
            log("INFO", "c.m.controller.ApiController", f"POST /api/v1/orders - 201 Created ({random.randint(20,200)}ms)")
        elif r < 0.95:
            log("DEBUG", "c.m.repository.UserRepository", f"Executing query: SELECT * FROM users WHERE id = {random.randint(1,9999)}")
        else:
            log("INFO", "c.m.messaging.KafkaConsumer", f"Consumed {random.randint(100,1000)} messages from topic orders-created in {random.randint(1,5)}s")

    # GC pressure events
    for _ in range(20):
        t += random.randint(1, 5)
        log("WARN", "o.s.b.a.HeapMemoryMonitor", f"Heap memory usage: {random.randint(85,99)}% ({random.randint(3500,4000)}MB/{random.randint(4000,4096)}MB)")

    return lines


# ---------------------------------------------------------------------------
# KernelErrors
# ---------------------------------------------------------------------------

def generate_kernel_errors(base: datetime) -> list[str]:
    lines: list[str] = []
    hostname = "prod-node-03"
    t = 100.0

    def klog(msg: str):
        nonlocal t
        t += random.uniform(0.001, 2.0)
        lines.append(f"{_ts_kernel(base, t)} {msg}")

    def syslog(facility: str, msg: str):
        nonlocal t
        t += random.uniform(0.1, 5.0)
        lines.append(f"{_ts_syslog(base, int(t))} {hostname} {facility}: {msg}")

    # Normal boot messages
    syslog("kernel", "Linux version 5.15.0-91-generic (buildd@lcy02-amd64-032)")
    syslog("kernel", "Command line: BOOT_IMAGE=/vmlinuz-5.15.0-91-generic root=/dev/mapper/vg0-root ro quiet splash")
    syslog("systemd[1]", "Started Journal Service.")
    syslog("systemd[1]", "Reached target Local File Systems.")

    for _ in range(250):
        r = random.random()
        if r < 0.10:
            # OOM killer
            pid = random.randint(1000, 65000)
            proc = random.choice(["java", "python3", "node", "mongod", "mysqld", "redis-server"])
            rss = random.randint(500000, 4000000)
            klog(f"Out of memory: Kill process {pid} ({proc}) score {random.randint(500,1000)} or sacrifice child")
            klog(f"Killed process {pid} ({proc}) total-vm:{rss}kB, anon-rss:{rss//2}kB, file-rss:0kB, shmem-rss:0kB")
            syslog("systemd[1]", f"{proc}.service: Main process exited, code=killed, status=9/KILL")
        elif r < 0.18:
            # Segfault
            pid = random.randint(1000, 65000)
            proc = random.choice(["nginx", "php-fpm", "apache2", "haproxy"])
            ip = f"{''.join(random.choices('0123456789abcdef', k=12))}"
            syslog("kernel", f"[{pid}]: segfault at {ip} ip {ip} sp 00007fff{random.randbytes(4).hex()} error 4 in {proc}[{ip}+{random.randint(1000,9999)}]")
        elif r < 0.26:
            # Disk I/O errors
            dev = random.choice(["sda", "sdb", "nvme0n1"])
            sector = random.randint(100000, 99999999)
            klog(f"blk_update_request: I/O error, dev {dev}, sector {sector} op 0x0:(READ) flags 0x0")
            klog(f"Buffer I/O error on dev {dev}1, logical block {sector//8}, async page read")
            syslog("smartd[892]", f"Device: /dev/{dev}, SMART Prefailure Attribute: 5 Reallocated_Sector_Ct changed from 100 to {random.randint(90,99)}")
        elif r < 0.33:
            # MCE (Machine Check Exception)
            klog(f"mce: [Hardware Error]: Machine check events logged")
            klog(f"mce: [Hardware Error]: CPU {random.randint(0,31)}: Machine Check: {random.randint(0,9)} Bank {random.randint(0,15)}: {random.randbytes(8).hex()}")
            klog(f"mce: [Hardware Error]: TSC {random.randbytes(6).hex()} ADDR {random.randbytes(6).hex()}")
        elif r < 0.40:
            # Filesystem errors
            dev = random.choice(["sda1", "sdb1", "dm-0"])
            syslog("kernel", f"EXT4-fs error (device {dev}): ext4_lookup:1698: inode #{random.randint(100000,999999)}: comm {random.choice(['ls', 'stat', 'find'])}: deleted inode referenced: {random.randint(100000,999999)}")
            syslog("kernel", f"EXT4-fs (device {dev}): Remounting filesystem read-only")
        elif r < 0.47:
            # Network errors
            iface = random.choice(["eth0", "ens5", "bond0"])
            syslog("kernel", f"{iface}: NIC Link is Down")
            t += 2
            syslog("kernel", f"{iface}: NIC Link is Up 10000 Mbps Full Duplex, Flow Control: Rx/Tx")
            syslog("kernel", f"NETDEV WATCHDOG: {iface} ({random.choice(['ixgbe', 'i40e', 'mlx5_core'])}): transmit queue {random.randint(0,7)} timed out")
        elif r < 0.53:
            # Kernel panic line
            klog("Kernel panic - not syncing: VFS: Unable to mount root fs on unknown-block(0,0)")
            klog("CPU: 0 PID: 1 Comm: swapper/0 Not tainted 5.15.0-91-generic #101-Ubuntu")
            klog("Call Trace:")
            klog(f" <TASK>")
            for fn in ["mount_root+0x127/0x160", "prepare_namespace+0x13c/0x170", "kernel_init+0x13/0x150"]:
                klog(f" {fn}")
            klog(f" </TASK>")
        elif r < 0.58:
            # Thermal throttling
            cpu = random.randint(0, 31)
            temp = random.randint(95, 110)
            klog(f"CPU{cpu}: Package temperature above threshold, cpu clock throttled (temperature: {temp}C)")
            klog(f"mce: CPU{cpu}: Core temperature above threshold, cpu clock throttled")
        elif r < 0.65:
            # USB/hardware errors
            syslog("kernel", f"usb {random.randint(1,4)}-{random.randint(1,4)}: device descriptor read/64, error -{random.choice([32, 71, 110])}")
            syslog("kernel", f"usb {random.randint(1,4)}-{random.randint(1,4)}: device not accepting address {random.randint(2,20)}, error -{random.choice([32, 71])}")
        elif r < 0.72:
            # EDAC memory errors
            klog(f"EDAC MC{random.randint(0,3)}: {random.randint(1,100)} CE error on CPU#{random.randint(0,3)} Channel#{random.randint(0,1)} DIMM#{random.randint(0,3)} (channel:{random.randint(0,1)} slot:{random.randint(0,3)})")
        elif r < 0.80:
            # Normal messages
            syslog("systemd[1]", f"Started {random.choice(['cron', 'sshd', 'rsyslog', 'systemd-journald'])} service.")
            syslog("CRON[{0}]".format(random.randint(1000, 65000)), "(root) CMD (/usr/bin/logrotate /etc/logrotate.conf)")
        elif r < 0.88:
            syslog("dhclient[{0}]".format(random.randint(500, 2000)), f"DHCPREQUEST for 10.0.{random.randint(0,255)}.{random.randint(1,254)} on eth0")
        else:
            syslog("kernel", f"audit: type=1400 audit({int(t)}.{random.randint(100,999)}:{random.randint(1,999)}): apparmor=\"ALLOWED\" operation=\"open\" profile=\"/usr/sbin/sshd\" name=\"/proc/{random.randint(1000,65000)}/status\"")

    return lines


# ---------------------------------------------------------------------------
# DatabaseErrors
# ---------------------------------------------------------------------------

def generate_database_errors(base: datetime) -> list[str]:
    lines: list[str] = []
    t = 0

    def pglog(level: str, msg: str):
        nonlocal t
        t += random.randint(0, 5)
        pid = random.randint(10000, 65000)
        lines.append(f"{_ts(base, t)} [{pid}] {level}:  {msg}")

    def mylog(msg: str):
        nonlocal t
        t += random.randint(0, 5)
        tid = random.randint(1, 200)
        lines.append(f"{_ts(base, t)} {tid} [Note] {msg}")

    # Normal startup
    pglog("LOG", "database system was shut down at " + _ts(base, t - 60))
    pglog("LOG", "database system is ready to accept connections")
    pglog("LOG", f"autovacuum launcher started")

    for _ in range(220):
        r = random.random()
        if r < 0.10:
            # Deadlocks
            pid1, pid2 = random.randint(10000, 65000), random.randint(10000, 65000)
            pglog("ERROR", f"deadlock detected")
            pglog("DETAIL", f"Process {pid1} waits for ShareLock on transaction {random.randint(100000,999999)}; blocked by process {pid2}.")
            pglog("DETAIL", f"Process {pid2} waits for ShareLock on transaction {random.randint(100000,999999)}; blocked by process {pid1}.")
            pglog("HINT", "See server log for query details.")
            pglog("CONTEXT", f"while updating tuple ({random.randint(0,999)},{random.randint(1,50)}) in relation \"{random.choice(['orders', 'inventory', 'users'])}\"")
        elif r < 0.18:
            # Connection refused / too many connections
            pglog("FATAL", f"sorry, too many clients already (max_connections={random.choice([100, 200, 500])})")
            pglog("LOG", f"connection received: host=10.0.{random.randint(0,255)}.{random.randint(1,254)} port={random.randint(30000,65000)}")
        elif r < 0.25:
            # Replication lag
            pglog("WARNING", f"replication lag is {random.randint(30, 300)} seconds on standby server")
            pglog("LOG", f"standby \"{random.choice(['replica-1', 'replica-2', 'replica-3'])}\" is now {random.randint(50,500)} MB behind")
            pglog("WARNING", f"canceling statement due to conflict with recovery on standby")
        elif r < 0.33:
            # Slow queries
            duration = random.randint(5000, 120000)
            table = random.choice(["orders", "events", "audit_log", "users", "sessions"])
            pglog("LOG", f"duration: {duration}.{random.randint(100,999)} ms  statement: SELECT * FROM {table} WHERE created_at > now() - interval '{random.randint(1,30)} days'")
            if duration > 30000:
                pglog("WARNING", f"terminating statement due to statement_timeout after {duration}ms")
        elif r < 0.40:
            # Disk full
            pglog("PANIC", f"could not write to file \"pg_wal/{random.randbytes(8).hex()}\": No space left on device")
            pglog("LOG", "server process (PID {0}) was terminated by signal 6: Aborted".format(random.randint(10000, 65000)))
            pglog("LOG", "terminating any other active server processes")
        elif r < 0.47:
            # Corruption
            pglog("ERROR", f"invalid page in block {random.randint(0, 999999)} of relation base/{random.randint(10000,99999)}/{random.randint(10000,99999)}")
            pglog("CONTEXT", f"automatic vacuum of table \"{random.choice(['mydb.public.events', 'mydb.public.orders'])}\"")
        elif r < 0.53:
            # MySQL-style errors
            mylog(f"InnoDB: Error: unable to create temporary file in /tmp; errno: {random.choice([28, 13])}")
            mylog(f"[ERROR] InnoDB: Cannot allocate {random.randint(128, 1024)} MB for the buffer pool")
        elif r < 0.58:
            # Auth failures
            pglog("FATAL", f"password authentication failed for user \"{random.choice(['app_user', 'analytics', 'readonly'])}\"")
            pglog("DETAIL", f"Connection matched pg_hba.conf line {random.randint(80,100)}: \"host all all 10.0.0.0/8 md5\"")
        elif r < 0.63:
            # Lock timeouts
            pglog("ERROR", f"canceling statement due to lock timeout")
            pglog("STATEMENT", f"ALTER TABLE {random.choice(['orders', 'users'])} ADD COLUMN IF NOT EXISTS new_col text")
        elif r < 0.70:
            # Checkpoint warnings
            pglog("LOG", f"checkpoint starting: time")
            pglog("LOG", f"checkpoint complete: wrote {random.randint(5000,50000)} buffers ({random.randint(10,95)}%); {random.randint(1,10)} WAL file(s) added")
            pglog("WARNING", f"checkpoints are occurring too frequently ({random.randint(15,55)} seconds apart)")
        elif r < 0.80:
            # Normal queries
            pglog("LOG", f"duration: {random.randint(1,50)}.{random.randint(100,999)} ms  statement: SELECT 1")
        elif r < 0.88:
            pglog("LOG", f"connection authorized: user={random.choice(['app', 'analytics'])} database={random.choice(['mydb', 'analytics_db'])}")
        else:
            pglog("LOG", f"automatic analyze of table \"mydb.public.{random.choice(['events', 'users', 'orders'])}\"")

    return lines


# ---------------------------------------------------------------------------
# WebServer
# ---------------------------------------------------------------------------

def generate_web_server(base: datetime) -> list[str]:
    lines: list[str] = []
    t = 0

    def nginx_error(level: str, msg: str):
        nonlocal t
        t += random.randint(0, 3)
        pid = random.randint(1000, 9999)
        tid = random.randint(1, 64)
        ts = _ts(base, t)
        lines.append(f"{ts} [error] {pid}#{tid}: *{random.randint(100,99999)} {msg}")

    def nginx_access(status: int, method: str, path: str, upstream_time: float | None = None):
        nonlocal t
        t += random.randint(0, 2)
        ts_str = (base + timedelta(seconds=t)).strftime("%d/%b/%Y:%H:%M:%S +0000")
        ip = f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
        size = random.randint(0, 50000)
        req_time = random.uniform(0.001, 30.0) if upstream_time else random.uniform(0.001, 0.5)
        ut_str = f" upstream_response_time={upstream_time:.3f}" if upstream_time else ""
        lines.append(f'{ip} - - [{ts_str}] "{method} {path} HTTP/1.1" {status} {size} request_time={req_time:.3f}{ut_str}')

    def apache_error(level: str, msg: str):
        nonlocal t
        t += random.randint(0, 3)
        ts_str = (base + timedelta(seconds=t)).strftime("%a %b %d %H:%M:%S.%f %Y")
        lines.append(f"[{ts_str}] [{level}] [pid {random.randint(1000,9999)}] {msg}")

    # Normal startup
    lines.append(f"{_ts(base, 0)} [notice] 1#1: nginx/1.25.3")
    lines.append(f"{_ts(base, 0)} [notice] 1#1: built by gcc 12.2.0")
    lines.append(f"{_ts(base, 0)} [notice] 1#1: using the \"epoll\" event method")
    lines.append(f"{_ts(base, 0)} [notice] 1#1: start worker processes")

    for _ in range(250):
        r = random.random()
        if r < 0.10:
            # 502 Bad Gateway
            upstream = f"10.0.{random.randint(1,10)}.{random.randint(1,254)}:{random.choice([8080, 3000, 5000])}"
            nginx_error("error", f"connect() failed (111: Connection refused) while connecting to upstream, client: {random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}, upstream: \"http://{upstream}/api/v1/data\"")
            nginx_access(502, "GET", f"/api/v1/data?page={random.randint(1,100)}")
        elif r < 0.18:
            # 504 Gateway Timeout
            upstream = f"10.0.{random.randint(1,10)}.{random.randint(1,254)}:{random.choice([8080, 3000])}"
            nginx_error("error", f"upstream timed out (110: Connection timed out) while reading response header from upstream, client: {random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}, upstream: \"http://{upstream}/api/v1/search\"")
            nginx_access(504, "POST", "/api/v1/search", upstream_time=60.0)
        elif r < 0.25:
            # 503 Service Unavailable
            nginx_error("error", f"no live upstreams while connecting to upstream, server: api.example.com")
            nginx_access(503, "GET", f"/health")
        elif r < 0.32:
            # SSL errors
            nginx_error("crit", f"SSL_do_handshake() failed (SSL: error:{random.randbytes(4).hex()}:SSL routines:ssl3_read_bytes:sslv3 alert certificate unknown)")
            nginx_error("error", f"peer closed connection in SSL handshake while SSL handshaking, client: {random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}")
        elif r < 0.38:
            # Connection resets
            nginx_error("error", f"recv() failed (104: Connection reset by peer) while reading response header from upstream")
            nginx_access(502, random.choice(["GET", "POST"]), f"/api/v1/{random.choice(['users', 'orders', 'data'])}")
        elif r < 0.44:
            # Apache 500 errors
            apache_error("error", f"AH01071: Got error 'PHP message: PHP Fatal error:  Allowed memory size of {random.choice([134217728, 268435456])} bytes exhausted (tried to allocate {random.randint(10000,99999)} bytes)")
            apache_error("error", f"AH01215: PHP Fatal error:  Uncaught Error: Call to undefined function {random.choice(['mysql_connect', 'split', 'ereg'])}()")
        elif r < 0.50:
            # Rate limiting
            nginx_access(429, "POST", f"/api/v1/login")
            nginx_error("error", f"limiting requests, excess: {random.randint(10,100)}.{random.randint(100,999)} by zone \"api_rate_limit\"")
        elif r < 0.56:
            # Client errors
            nginx_access(413, "POST", "/api/v1/upload", upstream_time=0.0)
            nginx_error("error", f"client intended to send too large body: {random.randint(10,500)} MB")
        elif r < 0.62:
            # Worker crashes
            lines.append(f"{_ts(base, t)} [alert] 1#1: worker process {random.randint(10,50)} exited on signal {random.choice([11, 6, 9])}")
            lines.append(f"{_ts(base, t)} [notice] 1#1: start worker process {random.randint(51,100)}")
        elif r < 0.75:
            # Normal 200 responses
            path = random.choice(["/", "/api/v1/users", "/api/v1/data", "/static/app.js", "/health"])
            nginx_access(200, "GET", path)
        elif r < 0.85:
            # Normal 301/304 responses
            nginx_access(random.choice([301, 304]), "GET", random.choice(["/old-page", "/assets/style.css"]))
        else:
            # Normal 404 (not errors in this context)
            nginx_access(404, "GET", f"/{random.choice(['favicon.ico', 'robots.txt', '.env', 'wp-admin'])}")

    return lines


# ---------------------------------------------------------------------------
# Microservices
# ---------------------------------------------------------------------------

def generate_microservices(base: datetime) -> list[str]:
    lines: list[str] = []
    t = 0

    services = ["user-svc", "order-svc", "payment-svc", "inventory-svc", "notification-svc", "gateway"]
    namespaces = ["production", "staging"]

    def kube_event(kind: str, name: str, namespace: str, msg: str):
        nonlocal t
        t += random.randint(0, 5)
        ts = _ts(base, t)
        lines.append(f"{ts} EVENT {kind}/{name} in {namespace}: {msg}")

    def svc_log(service: str, level: str, msg: str):
        nonlocal t
        t += random.randint(0, 3)
        ts = _ts(base, t)
        pod_id = f"{service}-{random.randbytes(3).hex()}-{random.randbytes(2).hex()}"
        lines.append(f"{ts} [{level:>5s}] [{pod_id}] {msg}")

    # Normal startup
    svc_log("gateway", "INFO", "Starting gateway service v3.2.1")
    svc_log("gateway", "INFO", "Listening on :8080")
    svc_log("user-svc", "INFO", "Connected to database")
    svc_log("order-svc", "INFO", "Kafka consumer group joined: order-events")

    for _ in range(200):
        r = random.random()
        svc = random.choice(services)

        if r < 0.10:
            # CrashLoopBackOff
            pod = f"{svc}-{random.randbytes(3).hex()}-{random.randbytes(2).hex()}"
            kube_event("Pod", pod, "production", f"Back-off restarting failed container (restart count: {random.randint(3,50)})")
            kube_event("Pod", pod, "production", f"Container {svc} failed liveness probe: HTTP probe failed with statuscode: 503")
            svc_log(svc, "FATAL", f"panic: runtime error: index out of range [{random.randint(0,10)}] with length {random.randint(0,5)}")
            lines.append(f"goroutine {random.randint(1,100)} [running]:")
            lines.append(f"main.(*Server).handleRequest(0xc000{random.randbytes(3).hex()}, {{0xc000{random.randbytes(3).hex()}, 0x{random.randbytes(2).hex()}}})")
            lines.append(f"\t/app/server.go:{random.randint(50,300)}")
        elif r < 0.18:
            # gRPC errors
            target_svc = random.choice([s for s in services if s != svc])
            svc_log(svc, "ERROR", f"gRPC call to {target_svc} failed: rpc error: code = DeadlineExceeded desc = context deadline exceeded (timeout: {random.choice(['5s', '10s', '30s'])})")
        elif r < 0.25:
            # Circuit breaker
            target_svc = random.choice([s for s in services if s != svc])
            svc_log(svc, "WARN", f"circuit breaker '{target_svc}' state changed: CLOSED -> OPEN (failures: {random.randint(5,20)}, threshold: 5)")
            svc_log(svc, "ERROR", f"request to {target_svc} rejected: circuit breaker is open")
        elif r < 0.32:
            # OOMKilled
            pod = f"{svc}-{random.randbytes(3).hex()}-{random.randbytes(2).hex()}"
            kube_event("Pod", pod, "production", f"OOMKilled: Container {svc} exceeded memory limit ({random.choice(['256Mi', '512Mi', '1Gi'])})")
            kube_event("Pod", pod, "production", f"Pulling image: registry.internal/myorg/{svc}:v{random.randint(1,5)}.{random.randint(0,20)}.{random.randint(0,99)}")
        elif r < 0.38:
            # Health check failures
            svc_log(svc, "WARN", f"readiness probe failed: connection to database timed out after 5s")
            kube_event("Pod", f"{svc}-{random.randbytes(3).hex()}-{random.randbytes(2).hex()}", "production",
                       f"Readiness probe failed: HTTP probe failed with statuscode: 503")
        elif r < 0.44:
            # Deployment issues
            dep = f"{svc}"
            kube_event("Deployment", dep, "production", f"Scaled up replica set {dep}-{random.randbytes(3).hex()} to {random.randint(1,5)}")
            kube_event("Deployment", dep, "production", f"deadline exceeded: progress deadline exceeded (timeout: 600s)")
            svc_log(svc, "ERROR", f"failed to pull image: registry.internal/myorg/{svc}:v99.0.0: not found")
        elif r < 0.50:
            # Service mesh errors
            svc_log("gateway", "ERROR", f"upstream connect error or disconnect/reset before headers. reset reason: connection failure, transport failure reason: delayed connect error: {random.choice([111, 113])}")
        elif r < 0.56:
            # Resource quota
            kube_event("Pod", f"{svc}-{random.randbytes(3).hex()}-{random.randbytes(2).hex()}", "production",
                       f"FailedScheduling: 0/{random.randint(5,20)} nodes are available: {random.randint(3,10)} Insufficient cpu, {random.randint(2,8)} Insufficient memory")
        elif r < 0.62:
            # Config/secret errors
            svc_log(svc, "FATAL", f"failed to load config: secret \"production/{svc}-secrets\" not found")
            kube_event("Pod", f"{svc}-{random.randbytes(3).hex()}-{random.randbytes(2).hex()}", "production",
                       f"Error: secret \"{svc}-secrets\" not found")
        elif r < 0.72:
            # Normal request logs
            svc_log(svc, "INFO", f"request completed: method=GET path=/api/v1/{random.choice(['users', 'orders', 'items'])} status=200 duration={random.randint(1,50)}ms")
        elif r < 0.80:
            svc_log(svc, "INFO", f"processed event: type={random.choice(['order.created', 'user.updated', 'payment.completed'])} id={random.randbytes(4).hex()}")
        else:
            svc_log(svc, "DEBUG", f"health check passed: {svc} is healthy")

    return lines


# ---------------------------------------------------------------------------
# SecurityEvents
# ---------------------------------------------------------------------------

def generate_security_events(base: datetime) -> list[str]:
    lines: list[str] = []
    t = 0
    hostname = "bastion-01"

    def syslog(facility: str, msg: str):
        nonlocal t
        t += random.randint(0, 5)
        ts = _ts_syslog(base, t)
        lines.append(f"{ts} {hostname} {facility}: {msg}")

    def authlog(msg: str):
        syslog("sshd[{0}]".format(random.randint(1000, 65000)), msg)

    def fwlog(msg: str):
        nonlocal t
        t += random.randint(0, 3)
        ts = _ts_syslog(base, t)
        lines.append(f"{ts} {hostname} kernel: [{random.randint(10000,99999)}.{random.randint(100,999)}] [UFW BLOCK] {msg}")

    attacker_ips = [f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}" for _ in range(5)]
    usernames = ["root", "admin", "ubuntu", "test", "oracle", "postgres", "deploy", "git"]

    # Normal auth
    syslog("systemd[1]", "Started OpenBSD Secure Shell server.")
    authlog(f"Server listening on 0.0.0.0 port 22.")

    for _ in range(180):
        r = random.random()

        if r < 0.15:
            # SSH brute force
            ip = random.choice(attacker_ips)
            user = random.choice(usernames)
            for _ in range(random.randint(3, 8)):
                authlog(f"Failed password for {'invalid user ' if random.random() < 0.5 else ''}{user} from {ip} port {random.randint(30000,65000)} ssh2")
            authlog(f"error: maximum authentication attempts exceeded for {user} from {ip} port {random.randint(30000,65000)} ssh2 [preauth]")
            syslog(f"fail2ban.actions[{random.randint(1000,9999)}]", f"NOTICE [sshd] Ban {ip}")
        elif r < 0.22:
            # Certificate expiry
            syslog("nginx", f"SSL: error:0A000086:SSL routines::certificate verify failed (certificate has expired)")
            syslog("certbot", f"Certificate for domain {random.choice(['api.example.com', 'www.example.com', 'auth.example.com'])} expires in {random.randint(0,5)} days!")
        elif r < 0.30:
            # Firewall drops
            ip = random.choice(attacker_ips)
            fwlog(f"IN=eth0 OUT= MAC={random.randbytes(6).hex()} SRC={ip} DST=10.0.1.{random.randint(1,254)} LEN=40 TOS=0x00 PREC=0x00 TTL=240 PROTO=TCP SPT={random.randint(1024,65000)} DPT={random.choice([22, 80, 443, 3306, 5432, 6379, 8080, 9200])} WINDOW=1024 SYN")
        elif r < 0.37:
            # Suspicious processes
            syslog("audit[{0}]".format(random.randint(1000, 65000)),
                   f"type=EXECVE msg=audit({int(t)}.{random.randint(100,999)}:{random.randint(1,999)}): argc={random.randint(2,5)} a0=\"{random.choice(['/usr/bin/curl', '/usr/bin/wget', '/bin/bash', '/usr/bin/nc'])}\" a1=\"{random.choice(['-s', '-O', '-e', '-c'])}\" a2=\"{random.choice(['http://malicious.example.com/shell.sh', '/dev/tcp/10.0.0.1/4444', 'base64 -d'])}\""
                   )
            syslog(f"audit[{random.randint(1000,65000)}]", f"type=ANOMALY_RESP msg=audit({int(t)}.{random.randint(100,999)}): op=PAM:bad_ident acct=\"?\" exe=\"/usr/sbin/sshd\" hostname=? addr={random.choice(attacker_ips)} terminal=ssh res=failed")
        elif r < 0.44:
            # Privilege escalation attempts
            user = random.choice(["www-data", "nobody", "daemon"])
            syslog(f"sudo[{random.randint(1000,65000)}]", f"  {user} : user NOT in sudoers ; TTY=pts/{random.randint(0,5)} ; PWD=/tmp ; USER=root ; COMMAND={random.choice(['/bin/bash', '/bin/sh', '/usr/bin/id'])}")
        elif r < 0.50:
            # File integrity
            syslog("aide[{0}]".format(random.randint(1000, 9999)),
                   f"AIDE found differences between database and filesystem!!")
            syslog("aide[{0}]".format(random.randint(1000, 9999)),
                   f"changed: /etc/{random.choice(['passwd', 'shadow', 'sudoers', 'ssh/sshd_config', 'crontab'])}")
        elif r < 0.56:
            # PAM auth failures
            syslog(f"sshd[{random.randint(1000,65000)}]",
                   f"pam_unix(sshd:auth): authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost={random.choice(attacker_ips)}")
        elif r < 0.62:
            # Port scanning detection
            ip = random.choice(attacker_ips)
            syslog("portsentry[{0}]".format(random.randint(1000, 9999)),
                   f"attackalert: Connect from host: {ip} to TCP port: {random.randint(1,1024)}")
            syslog("portsentry[{0}]".format(random.randint(1000, 9999)),
                   f"attackalert: Host {ip} has been blocked via iptables")
        elif r < 0.70:
            # Successful auth (still noteworthy)
            ip = f"10.0.{random.randint(0,10)}.{random.randint(1,254)}"
            user = random.choice(["deploy", "admin", "ops"])
            authlog(f"Accepted publickey for {user} from {ip} port {random.randint(30000,65000)} ssh2: RSA SHA256:{random.randbytes(16).hex()}")
            authlog(f"pam_unix(sshd:session): session opened for user {user}(uid={random.randint(1000,2000)}) by (uid=0)")
        elif r < 0.80:
            # Normal cron
            syslog(f"CRON[{random.randint(1000,65000)}]", f"(root) CMD (/usr/bin/logrotate /etc/logrotate.conf)")
        elif r < 0.88:
            # Normal systemd
            syslog("systemd[1]", f"Started {random.choice(['cron.service', 'logrotate.service', 'apt-daily.service'])}.")
        else:
            # NTP sync
            syslog("ntpd[{0}]".format(random.randint(500, 2000)), f"adjusting local clock by {random.uniform(-0.5, 0.5):.6f}s")

    return lines


# ---------------------------------------------------------------------------
# Registry & main
# ---------------------------------------------------------------------------

GENERATORS: dict[str, callable] = {
    "JavaApp": generate_java_app,
    "KernelErrors": generate_kernel_errors,
    "DatabaseErrors": generate_database_errors,
    "WebServer": generate_web_server,
    "Microservices": generate_microservices,
    "SecurityEvents": generate_security_events,
}


def main() -> None:
    LOGHUB_DIR.mkdir(parents=True, exist_ok=True)
    base_time = datetime(2025, 6, 15, 10, 0, 0)

    print(f"Generating {len(GENERATORS)} error-heavy log files ...\n")

    for name, gen_fn in GENERATORS.items():
        lines = gen_fn(base_time)
        path = LOGHUB_DIR / f"{name}_2k.log"
        with open(path, "w") as f:
            for line in lines:
                f.write(line.rstrip("\n") + "\n")

        print(f"  {name:20s} → {len(lines):4d} lines  ({path.name})")

    print(f"\nDone. Files written to {LOGHUB_DIR}/")


if __name__ == "__main__":
    main()
