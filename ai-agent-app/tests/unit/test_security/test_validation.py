"""Tests for security validation utilities."""

from ai_agent.security.validation import SecurityValidator


class TestSecurityValidator:
    """Test security validation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = SecurityValidator()

    def test_validate_email_valid(self):
        """Test valid email validation."""
        valid_emails = [
            "test@example.com",
            "user.name@domain.co.uk",
            "user+tag@example.org",
            "test123@test-domain.com",
        ]

        for email in valid_emails:
            assert self.validator.validate_email(email) is True

    def test_validate_email_invalid(self):
        """Test invalid email validation."""
        invalid_emails = [
            "",
            "invalid-email",
            "@example.com",
            "test@",
            "test..test@example.com",
            "test@.com",
        ]

        for email in invalid_emails:
            assert self.validator.validate_email(email) is False

    def test_validate_password_strong(self):
        """Test strong password validation."""
        result = self.validator.validate_password("StrongPass123!")
        assert result["is_valid"] is True
        assert result["score"] >= 4
        assert len(result["issues"]) == 0

    def test_validate_password_weak(self):
        """Test weak password validation."""
        result = self.validator.validate_password("weak")
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0

    def test_validate_username_valid(self):
        """Test valid username validation."""
        result = self.validator.validate_username("valid_user123")
        assert result["is_valid"] is True
        assert len(result["issues"]) == 0

    def test_validate_username_invalid(self):
        """Test invalid username validation."""
        result = self.validator.validate_username("a")  # Too short
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0

    def test_detect_sql_injection_positive(self):
        """Test SQL injection detection with malicious input."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM users",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --",
            "1' OR 1=1--",
            "admin'--",
            "' OR 1=1#",
        ]

        for malicious_input in malicious_inputs:
            assert self.validator.detect_sql_injection(malicious_input) is True

    def test_detect_sql_injection_negative(self):
        """Test SQL injection detection with safe input."""
        safe_inputs = [
            "normal text",
            "user@example.com",
            "password123",
            "SELECT is a word in English",
            "This is a normal query",
        ]

        for safe_input in safe_inputs:
            assert self.validator.detect_sql_injection(safe_input) is False

    def test_detect_xss_positive(self):
        """Test XSS detection with malicious input."""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<iframe src='javascript:alert(\"xss\")'></iframe>",
            "expression(alert('xss'))",
        ]

        for malicious_input in malicious_inputs:
            assert self.validator.detect_xss(malicious_input) is True

    def test_detect_xss_negative(self):
        """Test XSS detection with safe input."""
        safe_inputs = [
            "normal text",
            "This is a <b>bold</b> text",
            "https://example.com",
            "user@example.com",
        ]

        for safe_input in safe_inputs:
            assert self.validator.detect_xss(safe_input) is False

    def test_detect_command_injection_positive(self):
        """Test command injection detection with malicious input."""
        malicious_inputs = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "`whoami`",
            "&& ls -la",
            "|| cat /etc/shadow",
            "$(id)",
        ]

        for malicious_input in malicious_inputs:
            assert self.validator.detect_command_injection(malicious_input) is True

    def test_detect_command_injection_negative(self):
        """Test command injection detection with safe input."""
        safe_inputs = [
            "normal text",
            "user@example.com",
            "password123",
            "This is a normal string",
        ]

        for safe_input in safe_inputs:
            assert self.validator.detect_command_injection(safe_input) is False

    def test_detect_path_traversal_positive(self):
        """Test path traversal detection with malicious input."""
        malicious_inputs = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "..%2f..%2f..%2fetc%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd",
        ]

        for malicious_input in malicious_inputs:
            assert self.validator.detect_path_traversal(malicious_input) is True

    def test_detect_path_traversal_negative(self):
        """Test path traversal detection with safe input."""
        safe_inputs = [
            "normal/path/file.txt",
            "images/photo.jpg",
            "documents/report.pdf",
        ]

        for safe_input in safe_inputs:
            assert self.validator.detect_path_traversal(safe_input) is False

    def test_validate_input_security_comprehensive(self):
        """Test comprehensive input security validation."""
        # Test safe input
        result = self.validator.validate_input_security("safe input", "general")
        assert result["is_valid"] is True
        assert result["is_safe"] is True
        assert len(result["threats_detected"]) == 0

        # Test malicious input
        result = self.validator.validate_input_security(
            "'; DROP TABLE users; --", "general"
        )
        assert result["is_safe"] is False
        assert "SQL injection detected" in result["threats_detected"]

        # Test XSS input
        result = self.validator.validate_input_security(
            "<script>alert('xss')</script>", "general"
        )
        assert result["is_safe"] is False
        assert "XSS detected" in result["threats_detected"]

    def test_sanitize_input(self):
        """Test input sanitization."""
        # Test normal input
        result = self.validator.sanitize_input("normal input")
        assert result == "normal input"

        # Test input with control characters
        result = self.validator.sanitize_input("input\x00with\x01control\x02chars")
        assert "\x00" not in result
        assert "\x01" not in result
        assert "\x02" not in result

        # Test long input truncation
        long_input = "a" * 2000
        result = self.validator.sanitize_input(long_input, max_length=1000)
        assert len(result) == 1000

    def test_validate_json(self):
        """Test JSON validation."""
        # Valid JSON
        assert self.validator.validate_json('{"key": "value"}') is True
        assert self.validator.validate_json("[1, 2, 3]") is True

        # Invalid JSON
        assert self.validator.validate_json('{"key": "value"') is False
        assert self.validator.validate_json("invalid json") is False

    def test_validate_api_key_format(self):
        """Test API key format validation."""
        # Valid API key
        valid_key = "sk-" + "a" * 32
        assert self.validator.validate_api_key_format(valid_key) is True

        # Invalid API key (too short)
        invalid_key = "sk-" + "a" * 10
        assert self.validator.validate_api_key_format(invalid_key) is False

        # Invalid API key (invalid characters)
        invalid_key = "sk-" + "a" * 30 + "!"
        assert self.validator.validate_api_key_format(invalid_key) is False

    def test_validate_jwt_format(self):
        """Test JWT format validation."""
        # Valid JWT format (3 parts separated by dots)
        valid_jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        assert self.validator.validate_jwt_format(valid_jwt) is True

        # Invalid JWT format
        assert self.validator.validate_jwt_format("invalid.jwt") is False
        assert self.validator.validate_jwt_format("not.a.valid.jwt.format") is False
        assert self.validator.validate_jwt_format("") is False
