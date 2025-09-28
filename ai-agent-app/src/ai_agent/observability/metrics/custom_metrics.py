"""Custom metrics collection for business logic."""

from typing import Any
from dataclasses import dataclass
from prometheus_client import Counter, Gauge, Histogram


@dataclass
class BusinessMetrics:
    """Business metrics configuration."""

    # Request metrics
    request_count: Counter
    request_duration: Histogram
    request_size: Histogram
    response_size: Histogram

    # Error metrics
    error_count: Counter
    error_rate: Gauge

    # Business logic metrics
    user_actions: Counter
    feature_usage: Counter
    conversion_events: Counter

    # Performance metrics
    cache_hits: Counter
    cache_misses: Counter
    database_queries: Counter
    external_api_calls: Counter


class CustomMetricsCollector:
    """Collector for custom business metrics."""

    def __init__(self, namespace: str = "ai_agent"):
        """Initialize custom metrics collector.

        Args:
            namespace: Prometheus namespace for metrics
        """
        self.namespace = namespace
        self._metrics: dict[str, Any] = {}
        self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """Initialize all custom metrics."""
        # Request metrics
        self._metrics["request_count"] = Counter(
            "requests_total",
            "Total number of requests",
            ["method", "endpoint", "status_code"],
            namespace=self.namespace,
        )

        self._metrics["request_duration"] = Histogram(
            "request_duration_seconds",
            "Request duration in seconds",
            ["method", "endpoint"],
            namespace=self.namespace,
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )

        self._metrics["request_size"] = Histogram(
            "request_size_bytes",
            "Request size in bytes",
            ["method", "endpoint"],
            namespace=self.namespace,
            buckets=(100, 1000, 10000, 100000, 1000000),
        )

        self._metrics["response_size"] = Histogram(
            "response_size_bytes",
            "Response size in bytes",
            ["method", "endpoint"],
            namespace=self.namespace,
            buckets=(100, 1000, 10000, 100000, 1000000),
        )

        # Error metrics
        self._metrics["error_count"] = Counter(
            "errors_total",
            "Total number of errors",
            ["error_type", "endpoint"],
            namespace=self.namespace,
        )

        self._metrics["error_rate"] = Gauge(
            "error_rate", "Current error rate", ["endpoint"], namespace=self.namespace
        )

        # Business logic metrics
        self._metrics["user_actions"] = Counter(
            "user_actions_total",
            "Total user actions",
            ["action_type", "user_id"],
            namespace=self.namespace,
        )

        self._metrics["feature_usage"] = Counter(
            "feature_usage_total",
            "Feature usage count",
            ["feature_name", "user_id"],
            namespace=self.namespace,
        )

        self._metrics["conversion_events"] = Counter(
            "conversion_events_total",
            "Conversion events",
            ["event_type", "user_id"],
            namespace=self.namespace,
        )

        # Performance metrics
        self._metrics["cache_hits"] = Counter(
            "cache_hits_total", "Cache hits", ["cache_name"], namespace=self.namespace
        )

        self._metrics["cache_misses"] = Counter(
            "cache_misses_total",
            "Cache misses",
            ["cache_name"],
            namespace=self.namespace,
        )

        self._metrics["database_queries"] = Counter(
            "database_queries_total",
            "Database queries",
            ["query_type", "table"],
            namespace=self.namespace,
        )

        self._metrics["external_api_calls"] = Counter(
            "external_api_calls_total",
            "External API calls",
            ["service", "endpoint", "status"],
            namespace=self.namespace,
        )

    def get_business_metrics(self) -> BusinessMetrics:
        """Get business metrics configuration."""
        return BusinessMetrics(
            request_count=self._metrics["request_count"],
            request_duration=self._metrics["request_duration"],
            request_size=self._metrics["request_size"],
            response_size=self._metrics["response_size"],
            error_count=self._metrics["error_count"],
            error_rate=self._metrics["error_rate"],
            user_actions=self._metrics["user_actions"],
            feature_usage=self._metrics["feature_usage"],
            conversion_events=self._metrics["conversion_events"],
            cache_hits=self._metrics["cache_hits"],
            cache_misses=self._metrics["cache_misses"],
            database_queries=self._metrics["database_queries"],
            external_api_calls=self._metrics["external_api_calls"],
        )

    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        request_size: int | None = None,
        response_size: int | None = None,
    ) -> None:
        """Record request metrics.

        Args:
            method: HTTP method
            endpoint: API endpoint
            status_code: HTTP status code
            duration: Request duration in seconds
            request_size: Request size in bytes
            response_size: Response size in bytes
        """
        self._metrics["request_count"].labels(
            method=method, endpoint=endpoint, status_code=str(status_code)
        ).inc()

        self._metrics["request_duration"].labels(
            method=method, endpoint=endpoint
        ).observe(duration)

        if request_size is not None:
            self._metrics["request_size"].labels(
                method=method, endpoint=endpoint
            ).observe(request_size)

        if response_size is not None:
            self._metrics["response_size"].labels(
                method=method, endpoint=endpoint
            ).observe(response_size)

    def record_error(
        self,
        error_type: str,
        endpoint: str,
    ) -> None:
        """Record error metrics.

        Args:
            error_type: Type of error
            endpoint: API endpoint where error occurred
        """
        self._metrics["error_count"].labels(
            error_type=error_type, endpoint=endpoint
        ).inc()

    def record_user_action(
        self,
        action_type: str,
        user_id: str,
    ) -> None:
        """Record user action.

        Args:
            action_type: Type of user action
            user_id: User identifier
        """
        self._metrics["user_actions"].labels(
            action_type=action_type, user_id=user_id
        ).inc()

    def record_feature_usage(
        self,
        feature_name: str,
        user_id: str,
    ) -> None:
        """Record feature usage.

        Args:
            feature_name: Name of the feature
            user_id: User identifier
        """
        self._metrics["feature_usage"].labels(
            feature_name=feature_name, user_id=user_id
        ).inc()

    def record_conversion_event(
        self,
        event_type: str,
        user_id: str,
    ) -> None:
        """Record conversion event.

        Args:
            event_type: Type of conversion event
            user_id: User identifier
        """
        self._metrics["conversion_events"].labels(
            event_type=event_type, user_id=user_id
        ).inc()

    def record_cache_hit(self, cache_name: str) -> None:
        """Record cache hit.

        Args:
            cache_name: Name of the cache
        """
        self._metrics["cache_hits"].labels(cache_name=cache_name).inc()

    def record_cache_miss(self, cache_name: str) -> None:
        """Record cache miss.

        Args:
            cache_name: Name of the cache
        """
        self._metrics["cache_misses"].labels(cache_name=cache_name).inc()

    def record_database_query(
        self,
        query_type: str,
        table: str,
    ) -> None:
        """Record database query.

        Args:
            query_type: Type of query (SELECT, INSERT, UPDATE, DELETE)
            table: Database table name
        """
        self._metrics["database_queries"].labels(
            query_type=query_type, table=table
        ).inc()

    def record_external_api_call(
        self,
        service: str,
        endpoint: str,
        status: str,
    ) -> None:
        """Record external API call.

        Args:
            service: External service name
            endpoint: API endpoint
            status: Call status (success, error, timeout)
        """
        self._metrics["external_api_calls"].labels(
            service=service, endpoint=endpoint, status=status
        ).inc()

    def get_metric(self, metric_name: str) -> Any:
        """Get a specific metric by name.

        Args:
            metric_name: Name of the metric

        Returns:
            The metric object
        """
        return self._metrics.get(metric_name)

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all metrics.

        Returns:
            Dictionary of all metrics
        """
        return self._metrics.copy()
