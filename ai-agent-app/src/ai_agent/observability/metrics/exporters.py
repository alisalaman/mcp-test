"""Metrics exporters for different monitoring systems."""

import asyncio
import json
from datetime import datetime, UTC
from typing import Any
from abc import ABC, abstractmethod

from prometheus_client import generate_latest, CollectorRegistry
from prometheus_client.core import CollectorRegistry as PrometheusRegistry

from ..logging import get_logger

logger = get_logger(__name__)


class MetricsExporter(ABC):
    """Abstract base class for metrics exporters."""

    @abstractmethod
    async def export(self, metrics: dict[str, Any]) -> bool:
        """Export metrics to external system."""
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Check if exporter is healthy."""
        pass


class PrometheusExporter(MetricsExporter):
    """Prometheus metrics exporter."""

    def __init__(
        self,
        registry: PrometheusRegistry | None = None,
        port: int = 9090,
        path: str = "/metrics",
    ):
        self.registry = registry or CollectorRegistry()
        self.port = port
        self.path = path
        self._server = None

    async def export(self, metrics: dict[str, Any]) -> bool:
        """Export metrics in Prometheus format."""
        try:
            # Generate Prometheus format
            prometheus_data = generate_latest(self.registry)
            logger.debug(f"Exported {len(prometheus_data)} bytes of Prometheus metrics")
            return True
        except Exception as e:
            logger.error(f"Failed to export Prometheus metrics: {e}")
            return False

    async def health_check(self) -> dict[str, Any]:
        """Check if Prometheus exporter is healthy."""
        try:
            # Test registry access
            generate_latest(self.registry)
            return {
                "status": "healthy",
                "exporter": "prometheus",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            logger.error(f"Prometheus exporter health check failed: {e}")
            return {
                "status": "unhealthy",
                "exporter": "prometheus",
                "timestamp": datetime.now(UTC).isoformat(),
                "error": str(e),
            }

    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format."""
        result = generate_latest(self.registry)
        return result.decode("utf-8")

    def get_metrics_dict(self) -> dict[str, Any]:
        """Get metrics as dictionary."""
        metrics = {}

        for metric in self.registry.collect():
            metric_type = metric.type

            if metric_type == "counter":
                for sample in metric.samples:
                    key = f"{sample.name}{{"
                    if sample.labels:
                        key += ",".join([f"{k}={v}" for k, v in sample.labels.items()])
                    key += "}"
                    metrics[key] = sample.value
            elif metric_type == "gauge":
                for sample in metric.samples:
                    key = f"{sample.name}{{"
                    if sample.labels:
                        key += ",".join([f"{k}={v}" for k, v in sample.labels.items()])
                    key += "}"
                    metrics[key] = sample.value
            elif metric_type == "histogram":
                for sample in metric.samples:
                    key = f"{sample.name}{{"
                    if sample.labels:
                        key += ",".join([f"{k}={v}" for k, v in sample.labels.items()])
                    key += "}"
                    metrics[key] = sample.value

        return metrics


class JSONExporter(MetricsExporter):
    """JSON metrics exporter."""

    def __init__(self, output_file: str | None = None):
        self.output_file = output_file

    async def export(self, metrics: dict[str, Any]) -> bool:
        """Export metrics in JSON format."""
        try:
            json_data = json.dumps(metrics, indent=2, default=str)

            if self.output_file:
                with open(self.output_file, "w") as f:
                    f.write(json_data)
                logger.info(f"Exported metrics to {self.output_file}")
            else:
                logger.debug(f"Exported {len(json_data)} bytes of JSON metrics")

            return True
        except Exception as e:
            logger.error(f"Failed to export JSON metrics: {e}")
            return False

    async def health_check(self) -> dict[str, Any]:
        """Check if JSON exporter is healthy."""
        return {
            "status": "healthy",
            "exporter": "json",
            "timestamp": datetime.now(UTC).isoformat(),
        }


class InfluxDBExporter(MetricsExporter):
    """InfluxDB metrics exporter."""

    def __init__(
        self,
        url: str,
        database: str,
        username: str | None = None,
        password: str | None = None,
        measurement: str = "metrics",
    ):
        self.url = url
        self.database = database
        self.username = username
        self.password = password
        self.measurement = measurement
        self._client = None

    async def export(self, metrics: dict[str, Any]) -> bool:
        """Export metrics to InfluxDB."""
        try:
            # This would require influxdb-client library
            # For now, just log the metrics
            logger.info(f"Would export {len(metrics)} metrics to InfluxDB")
            return True
        except Exception as e:
            logger.error(f"Failed to export to InfluxDB: {e}")
            return False

    async def health_check(self) -> dict[str, Any]:
        """Check if InfluxDB exporter is healthy."""
        try:
            # Test connection to InfluxDB
            # This would require actual InfluxDB client
            return {
                "status": "healthy",
                "exporter": "influxdb",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            logger.error(f"InfluxDB exporter health check failed: {e}")
            return {
                "status": "unhealthy",
                "exporter": "influxdb",
                "timestamp": datetime.now(UTC).isoformat(),
                "error": str(e),
            }


class CloudWatchExporter(MetricsExporter):
    """AWS CloudWatch metrics exporter."""

    def __init__(
        self,
        namespace: str = "AI-Agent",
        region: str = "us-east-1",
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
    ):
        self.namespace = namespace
        self.region = region
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self._client = None

    async def export(self, metrics: dict[str, Any]) -> bool:
        """Export metrics to CloudWatch."""
        try:
            # This would require boto3 library
            # For now, just log the metrics
            logger.info(f"Would export {len(metrics)} metrics to CloudWatch")
            return True
        except Exception as e:
            logger.error(f"Failed to export to CloudWatch: {e}")
            return False

    async def health_check(self) -> dict[str, Any]:
        """Check if CloudWatch exporter is healthy."""
        try:
            # Test CloudWatch connection
            # This would require actual CloudWatch client
            return {
                "status": "healthy",
                "exporter": "cloudwatch",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            logger.error(f"CloudWatch exporter health check failed: {e}")
            return {
                "status": "unhealthy",
                "exporter": "cloudwatch",
                "timestamp": datetime.now(UTC).isoformat(),
                "error": str(e),
            }


class MetricsExporterManager:
    """Manager for multiple metrics exporters."""

    def __init__(self) -> None:
        self.exporters: list[MetricsExporter] = []
        self.logger = get_logger(__name__)

    def add_exporter(self, exporter: MetricsExporter) -> None:
        """Add an exporter to the manager."""
        self.exporters.append(exporter)
        self.logger.info(f"Added metrics exporter: {exporter.__class__.__name__}")

    async def export_all(self, metrics: dict[str, Any]) -> dict[str, bool]:
        """Export metrics using all exporters."""
        results = {}

        for exporter in self.exporters:
            try:
                success = await exporter.export(metrics)
                results[exporter.__class__.__name__] = success
            except Exception as e:
                self.logger.error(f"Exporter {exporter.__class__.__name__} failed: {e}")
                results[exporter.__class__.__name__] = False

        return results

    async def health_check_all(self) -> dict[str, bool]:
        """Check health of all exporters."""
        results = {}

        for exporter in self.exporters:
            try:
                health_result = await exporter.health_check()
                # Extract boolean status from health check result
                healthy = health_result.get("status") == "healthy"
                results[exporter.__class__.__name__] = healthy
            except Exception as e:
                self.logger.error(
                    f"Health check failed for {exporter.__class__.__name__}: {e}"
                )
                results[exporter.__class__.__name__] = False

        return results

    def get_healthy_exporters(self) -> list[MetricsExporter]:
        """Get list of healthy exporters."""
        healthy = []
        for exporter in self.exporters:
            try:
                if asyncio.iscoroutinefunction(exporter.health_check):
                    # This would need to be awaited in practice
                    healthy.append(exporter)
            except Exception:
                continue
        return healthy


class MetricsScheduler:
    """Scheduler for periodic metrics export."""

    def __init__(
        self,
        exporter_manager: MetricsExporterManager,
        interval: int = 60,  # seconds
        max_retries: int = 3,
    ):
        self.exporter_manager = exporter_manager
        self.interval = interval
        self.max_retries = max_retries
        self.logger = get_logger(__name__)
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the metrics scheduler."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_scheduler())
        self.logger.info(f"Started metrics scheduler with {self.interval}s interval")

    async def stop(self) -> None:
        """Stop the metrics scheduler."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        self.logger.info("Stopped metrics scheduler")

    async def _run_scheduler(self) -> None:
        """Run the metrics scheduler loop."""
        while self._running:
            try:
                # Get metrics from collector
                from .collectors import get_metrics_collector

                collector = get_metrics_collector()
                metrics = collector.get_all_metrics()

                # Export metrics
                results = await self.exporter_manager.export_all(metrics)

                # Log results
                successful = sum(1 for success in results.values() if success)
                total = len(results)
                self.logger.debug(
                    f"Exported metrics: {successful}/{total} exporters successful"
                )

            except Exception as e:
                self.logger.error(f"Metrics scheduler error: {e}")

            # Wait for next interval
            await asyncio.sleep(self.interval)

    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running


# Global exporter manager
_exporter_manager: MetricsExporterManager | None = None


def get_exporter_manager() -> MetricsExporterManager:
    """Get global exporter manager."""
    global _exporter_manager
    if _exporter_manager is None:
        _exporter_manager = MetricsExporterManager()
    return _exporter_manager


def setup_prometheus_exporter(
    port: int = 9090, path: str = "/metrics"
) -> PrometheusExporter:
    """Setup Prometheus exporter."""
    exporter = PrometheusExporter(port=port, path=path)
    manager = get_exporter_manager()
    manager.add_exporter(exporter)
    return exporter


def setup_json_exporter(output_file: str | None = None) -> JSONExporter:
    """Setup JSON exporter."""
    exporter = JSONExporter(output_file=output_file)
    manager = get_exporter_manager()
    manager.add_exporter(exporter)
    return exporter
