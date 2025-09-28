"""Metrics collectors and instrumentation."""

import time
from typing import Any
from functools import wraps
from contextlib import contextmanager

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
)
from prometheus_client.core import CollectorRegistry as PrometheusRegistry

from ..logging import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """Main metrics collector for the application."""

    def __init__(self, registry: PrometheusRegistry | None = None):
        self.registry = registry or PrometheusRegistry()
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}
        self._summaries: dict[str, Summary] = {}

        # Initialize default metrics
        self._setup_default_metrics()

    def _setup_default_metrics(self) -> None:
        """Setup default application metrics."""
        # HTTP request metrics
        self.http_requests_total = self.create_counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code"],
        )

        self.http_request_duration_seconds = self.create_histogram(
            "http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
            buckets=[
                0.005,
                0.01,
                0.025,
                0.05,
                0.075,
                0.1,
                0.25,
                0.5,
                0.75,
                1.0,
                2.5,
                5.0,
                7.5,
                10.0,
            ],
        )

        # Database metrics
        self.database_operations_total = self.create_counter(
            "database_operations_total",
            "Total database operations",
            ["operation", "table", "status"],
        )

        self.database_operation_duration_seconds = self.create_histogram(
            "database_operation_duration_seconds",
            "Database operation duration in seconds",
            ["operation", "table"],
        )

        # External service metrics
        self.external_service_requests_total = self.create_counter(
            "external_service_requests_total",
            "Total external service requests",
            ["service", "endpoint", "status"],
        )

        self.external_service_duration_seconds = self.create_histogram(
            "external_service_duration_seconds",
            "External service request duration in seconds",
            ["service", "endpoint"],
        )

        # Agent execution metrics
        self.agent_executions_total = self.create_counter(
            "agent_executions_total", "Total agent executions", ["agent_id", "status"]
        )

        self.agent_execution_duration_seconds = self.create_histogram(
            "agent_execution_duration_seconds",
            "Agent execution duration in seconds",
            ["agent_id"],
        )

        # System metrics
        self.active_connections = self.create_gauge(
            "active_connections", "Number of active connections"
        )

        self.memory_usage_bytes = self.create_gauge(
            "memory_usage_bytes", "Memory usage in bytes"
        )

        self.cpu_usage_percent = self.create_gauge(
            "cpu_usage_percent", "CPU usage percentage"
        )

    def create_counter(
        self, name: str, documentation: str, labelnames: list[str] | None = None
    ) -> Counter:
        """Create a counter metric."""
        if name in self._counters:
            return self._counters[name]

        counter = Counter(
            name, documentation, labelnames=labelnames or [], registry=self.registry
        )
        self._counters[name] = counter
        return counter

    def create_gauge(
        self, name: str, documentation: str, labelnames: list[str] | None = None
    ) -> Gauge:
        """Create a gauge metric."""
        if name in self._gauges:
            return self._gauges[name]

        gauge = Gauge(
            name, documentation, labelnames=labelnames or [], registry=self.registry
        )
        self._gauges[name] = gauge
        return gauge

    def create_histogram(
        self,
        name: str,
        documentation: str,
        labelnames: list[str] | None = None,
        buckets: list[float] | None = None,
    ) -> Histogram:
        """Create a histogram metric."""
        if name in self._histograms:
            return self._histograms[name]

        # Use default buckets if none provided
        default_buckets = [
            0.005,
            0.01,
            0.025,
            0.05,
            0.075,
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
            2.5,
            5.0,
            7.5,
            10.0,
        ]
        histogram = Histogram(
            name,
            documentation,
            labelnames=labelnames or [],
            buckets=buckets or default_buckets,
            registry=self.registry,
        )
        self._histograms[name] = histogram
        return histogram

    def create_summary(
        self, name: str, documentation: str, labelnames: list[str] | None = None
    ) -> Summary:
        """Create a summary metric."""
        if name in self._summaries:
            return self._summaries[name]

        summary = Summary(
            name, documentation, labelnames=labelnames or [], registry=self.registry
        )
        self._summaries[name] = summary
        return summary

    def get_counter(self, name: str) -> Counter | None:
        """Get a counter metric by name."""
        return self._counters.get(name)

    def get_gauge(self, name: str) -> Gauge | None:
        """Get a gauge metric by name."""
        return self._gauges.get(name)

    def get_histogram(self, name: str) -> Histogram | None:
        """Get a histogram metric by name."""
        return self._histograms.get(name)

    def get_summary(self, name: str) -> Summary | None:
        """Get a summary metric by name."""
        return self._summaries.get(name)

    def increment_counter(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        """Increment a counter metric."""
        counter = self.get_counter(name)
        if counter:
            if labels:
                counter.labels(**labels).inc(value)
            else:
                counter.inc(value)
        else:
            logger.warning(f"Counter metric '{name}' not found")

    def set_gauge(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Set a gauge metric value."""
        gauge = self.get_gauge(name)
        if gauge:
            if labels:
                gauge.labels(**labels).set(value)
            else:
                gauge.set(value)
        else:
            logger.warning(f"Gauge metric '{name}' not found")

    def observe_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Observe a histogram metric value."""
        histogram = self.get_histogram(name)
        if histogram:
            if labels:
                histogram.labels(**labels).observe(value)
            else:
                histogram.observe(value)
        else:
            logger.warning(f"Histogram metric '{name}' not found")

    def observe_summary(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Observe a summary metric value."""
        summary = self.get_summary(name)
        if summary:
            if labels:
                summary.labels(**labels).observe(value)
            else:
                summary.observe(value)
        else:
            logger.warning(f"Summary metric '{name}' not found")

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all metrics in a dictionary format."""
        metrics = {}

        # Add counters
        for name, counter in self._counters.items():
            metrics[name] = {
                "type": "counter",
                "labels": counter._labelnames,
            }

        # Add gauges
        for name, gauge in self._gauges.items():
            metrics[name] = {
                "type": "gauge",
                "labels": gauge._labelnames,
            }

        # Add histograms
        for name, histogram in self._histograms.items():
            metrics[name] = {
                "type": "histogram",
                "labels": histogram._labelnames,
            }

        # Add summaries
        for name, summary in self._summaries.items():
            metrics[name] = {"type": "summary", "labels": summary._labelnames}

        return metrics


def time_function(metric_name: str, labels: dict[str, str] | None = None) -> Any:
    """Decorator to time function execution."""

    def decorator(func: Any) -> Any:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                collector = get_metrics_collector()
                collector.observe_histogram(metric_name, duration, labels)

        return wrapper

    return decorator


def count_calls(metric_name: str, labels: dict[str, str] | None = None) -> Any:
    """Decorator to count function calls."""

    def decorator(func: Any) -> Any:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                result = func(*args, **kwargs)
                collector = get_metrics_collector()
                collector.increment_counter(metric_name, 1, labels)
                return result
            except Exception:
                collector = get_metrics_collector()
                error_labels = (labels or {}).copy()
                error_labels["status"] = "error"
                collector.increment_counter(metric_name, 1, error_labels)
                raise

        return wrapper

    return decorator


@contextmanager
def time_operation(metric_name: str, labels: dict[str, str] | None = None) -> Any:
    """Context manager to time operations."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        collector = get_metrics_collector()
        collector.observe_histogram(metric_name, duration, labels)


class MetricsMiddleware:
    """Middleware to automatically collect HTTP metrics."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector

    async def __call__(self, request: Any, call_next: Any) -> Any:
        """Process request and collect metrics."""
        start_time = time.time()

        # Extract request information
        method = request.method
        endpoint = request.url.path

        try:
            # Process request
            response = await call_next(request)

            # Record success metrics
            duration = time.time() - start_time
            self.metrics_collector.increment_counter(
                "http_requests_total",
                1,
                {
                    "method": method,
                    "endpoint": endpoint,
                    "status_code": str(response.status_code),
                },
            )
            self.metrics_collector.observe_histogram(
                "http_request_duration_seconds",
                duration,
                {"method": method, "endpoint": endpoint},
            )

            return response

        except Exception:
            # Record error metrics
            duration = time.time() - start_time
            self.metrics_collector.increment_counter(
                "http_requests_total",
                1,
                {"method": method, "endpoint": endpoint, "status_code": "500"},
            )
            self.metrics_collector.observe_histogram(
                "http_request_duration_seconds",
                duration,
                {"method": method, "endpoint": endpoint},
            )
            raise


# Global metrics collector instance
_metrics_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def setup_metrics(registry: PrometheusRegistry | None = None) -> MetricsCollector:
    """Setup global metrics collector."""
    global _metrics_collector
    _metrics_collector = MetricsCollector(registry)
    return _metrics_collector
