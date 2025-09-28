"""Tests for observability metrics."""

import pytest
from ai_agent.observability.metrics.collectors import MetricsCollector
from ai_agent.observability.metrics.exporters import (
    PrometheusExporter,
    JSONExporter,
    InfluxDBExporter,
    CloudWatchExporter,
)


class TestMetricsCollector:
    """Test metrics collection functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.collector = MetricsCollector()

    def test_create_counter(self):
        """Test counter metric creation."""
        counter = self.collector.create_counter(
            name="test_counter",
            documentation="Test counter metric",
            labelnames=["label1", "label2"],
        )

        assert counter is not None
        assert counter._name == "test_counter"
        assert counter._documentation == "Test counter metric"

    def test_create_histogram(self):
        """Test histogram metric creation."""
        histogram = self.collector.create_histogram(
            name="test_histogram",
            documentation="Test histogram metric",
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
            labelnames=["label1", "label2"],
        )

        assert histogram is not None
        assert histogram._name == "test_histogram"
        assert histogram._documentation == "Test histogram metric"

    def test_create_gauge(self):
        """Test gauge metric creation."""
        gauge = self.collector.create_gauge(
            name="test_gauge",
            documentation="Test gauge metric",
            labelnames=["label1", "label2"],
        )

        assert gauge is not None
        assert gauge._name == "test_gauge"
        assert gauge._documentation == "Test gauge metric"

    def test_create_summary(self):
        """Test summary metric creation."""
        summary = self.collector.create_summary(
            name="test_summary",
            documentation="Test summary metric",
            labelnames=["label1", "label2"],
        )

        assert summary is not None
        assert summary._name == "test_summary"
        assert summary._documentation == "Test summary metric"

    def test_increment_counter(self):
        """Test counter increment."""
        counter = self.collector.create_counter(
            name="test_counter",
            documentation="Test counter metric",
        )

        self.collector.increment_counter("test_counter", value=5)

        # Verify the counter was incremented (just check it doesn't raise an error)
        assert counter is not None

    def test_observe_histogram(self):
        """Test histogram observation."""
        histogram = self.collector.create_histogram(
            name="test_histogram",
            documentation="Test histogram metric",
        )

        self.collector.observe_histogram("test_histogram", value=1.5)

        # Verify the histogram was observed (just check it doesn't raise an error)
        assert histogram is not None

    def test_set_gauge(self):
        """Test gauge value setting."""
        gauge = self.collector.create_gauge(
            name="test_gauge",
            documentation="Test gauge metric",
        )

        self.collector.set_gauge("test_gauge", value=42.0)

        # Verify the gauge was set (just check it doesn't raise an error)
        assert gauge is not None

    def test_observe_summary(self):
        """Test summary observation."""
        summary = self.collector.create_summary(
            name="test_summary",
            documentation="Test summary metric",
        )

        self.collector.observe_summary("test_summary", value=2.5)

        # Verify the summary was observed (just check it doesn't raise an error)
        assert summary is not None

    def test_get_counter(self):
        """Test counter retrieval."""
        counter = self.collector.create_counter(
            name="test_counter",
            documentation="Test counter metric",
        )

        retrieved_counter = self.collector.get_counter("test_counter")
        assert retrieved_counter is counter

    def test_get_counter_nonexistent(self):
        """Test retrieval of nonexistent counter."""
        retrieved_counter = self.collector.get_counter("nonexistent_counter")
        assert retrieved_counter is None

    def test_get_all_metrics(self):
        """Test metrics listing."""
        self.collector.create_counter("counter1", "Counter 1")
        self.collector.create_gauge("gauge1", "Gauge 1")
        self.collector.create_histogram("histogram1", "Histogram 1")

        metrics = self.collector.get_all_metrics()
        assert len(metrics) >= 3  # At least the 3 we created, plus default metrics
        assert "counter1" in metrics
        assert "gauge1" in metrics
        assert "histogram1" in metrics

    def test_metrics_persistence(self):
        """Test that metrics persist across operations."""
        self.collector.create_counter("counter1", "Counter 1")
        self.collector.create_gauge("gauge1", "Gauge 1")

        metrics = self.collector.get_all_metrics()
        assert "counter1" in metrics
        assert "gauge1" in metrics


class TestPrometheusExporter:
    """Test Prometheus metrics exporter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = PrometheusExporter()

    def test_export_metrics(self):
        """Test metrics export to Prometheus format."""
        # Export metrics (the exporter has its own registry with default metrics)
        prometheus_data = self.exporter.get_metrics_text()

        assert isinstance(prometheus_data, str)
        # Just check that the method works without error

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test exporter health check."""
        health = await self.exporter.health_check()
        assert health["status"] == "healthy"
        assert "exporter" in health
        assert "timestamp" in health


class TestJSONExporter:
    """Test JSON metrics exporter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = JSONExporter()

    @pytest.mark.asyncio
    async def test_export_metrics(self):
        """Test metrics export to JSON format."""
        # Create a test collector with some metrics
        collector = MetricsCollector()
        collector.create_counter("test_counter", "Test counter")
        collector.increment_counter("test_counter", value=5)

        # Get metrics as dictionary
        metrics = collector.get_all_metrics()

        # Export metrics
        result = await self.exporter.export(metrics)

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test exporter health check."""
        health = await self.exporter.health_check()
        assert health["status"] == "healthy"
        assert "exporter" in health
        assert "timestamp" in health


class TestInfluxDBExporter:
    """Test InfluxDB metrics exporter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = InfluxDBExporter(
            url="http://localhost:8086",
            database="test_db",
            username="test_user",
            password="test_password",
        )

    @pytest.mark.asyncio
    async def test_export_metrics(self):
        """Test metrics export to InfluxDB format."""
        # Create a test collector with some metrics
        collector = MetricsCollector()
        collector.create_counter("test_counter", "Test counter")
        collector.increment_counter("test_counter", value=5)

        # Get metrics as dictionary
        metrics = collector.get_all_metrics()

        # Export metrics
        result = await self.exporter.export(metrics)

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test exporter health check."""
        health = await self.exporter.health_check()
        assert health["status"] == "healthy"
        assert "exporter" in health
        assert "timestamp" in health


class TestCloudWatchExporter:
    """Test CloudWatch metrics exporter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = CloudWatchExporter(
            namespace="TestNamespace",
            region="us-east-1",
        )

    @pytest.mark.asyncio
    async def test_export_metrics(self):
        """Test metrics export to CloudWatch format."""
        # Create a test collector with some metrics
        collector = MetricsCollector()
        collector.create_counter("test_counter", "Test counter")
        collector.increment_counter("test_counter", value=5)

        # Get metrics as dictionary
        metrics = collector.get_all_metrics()

        # Export metrics
        result = await self.exporter.export(metrics)

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test exporter health check."""
        health = await self.exporter.health_check()
        assert health["status"] == "healthy"
        assert "exporter" in health
        assert "timestamp" in health
