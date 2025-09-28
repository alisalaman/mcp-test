"""Security validation utilities."""

import re
import ipaddress
from typing import Any
from urllib.parse import urlparse

from ..observability.logging import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Validation error."""

    pass


class SecurityValidator:
    """Security validation utilities."""

    def __init__(self) -> None:
        self.logger = get_logger(__name__)

        # Pre-compile regex patterns for better performance
        self._email_pattern = re.compile(
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        )
        self._special_chars_pattern = re.compile(r"[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]")
        self._sequential_pattern = re.compile(
            r"(abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz|012|123|234|345|456|567|678|789|890)",
            re.IGNORECASE,
        )
        self._repeated_pattern = re.compile(r"(.)\1{2,}")

        # Common passwords set for O(1) lookup
        self._common_passwords = {
            "password",
            "123456",
            "123456789",
            "qwerty",
            "abc123",
            "password123",
            "admin",
            "letmein",
            "welcome",
            "monkey",
            "12345678",
            "password1",
            "123123",
            "qwerty123",
            "1q2w3e4r",
            "password12",
            "1234567890",
            "qwertyuiop",
            "asdfghjkl",
            "zxcvbnm",
        }

    def validate_email(self, email: str) -> bool:
        """Validate email format."""
        if not email or not isinstance(email, str):
            return False

        # Use pre-compiled pattern for better performance
        if not self._email_pattern.match(email):
            return False

        # Additional checks for invalid patterns
        # No consecutive dots in local part
        if ".." in email.split("@")[0]:
            return False

        # No leading or trailing dots in local part
        local_part = email.split("@")[0]
        if local_part.startswith(".") or local_part.endswith("."):
            return False

        # No consecutive dots in domain part
        domain_part = email.split("@")[1]
        if ".." in domain_part:
            return False

        return True

    def validate_password(self, password: str) -> dict[str, Any]:
        """Validate password strength."""
        result: dict[str, Any] = {
            "is_valid": True,
            "score": 0,
            "issues": [],
            "suggestions": [],
        }

        if not password or not isinstance(password, str):
            result["is_valid"] = False
            result["issues"].append("Password is required")
            return result

        # Length check
        if len(password) < 8:
            result["issues"].append("Password must be at least 8 characters long")
            result["is_valid"] = False
        elif len(password) >= 12:
            result["score"] += 1
        else:
            result["score"] += 0.5

        # Character variety checks - single pass through password
        has_upper = has_lower = has_digit = has_special = False
        for char in password:
            if char.isupper():
                has_upper = True
            elif char.islower():
                has_lower = True
            elif char.isdigit():
                has_digit = True
            elif self._special_chars_pattern.match(char):
                has_special = True

        if not has_upper:
            result["issues"].append(
                "Password must contain at least one uppercase letter"
            )
            result["is_valid"] = False
        else:
            result["score"] += 1

        if not has_lower:
            result["issues"].append(
                "Password must contain at least one lowercase letter"
            )
            result["is_valid"] = False
        else:
            result["score"] += 1

        if not has_digit:
            result["issues"].append("Password must contain at least one digit")
            result["is_valid"] = False
        else:
            result["score"] += 1

        if not has_special:
            result["issues"].append(
                "Password must contain at least one special character"
            )
            result["is_valid"] = False
        else:
            result["score"] += 1

        # Common password check - O(1) lookup
        if password.lower() in self._common_passwords:
            result["issues"].append("Password is too common")
            result["is_valid"] = False
            result["score"] -= 2

        # Sequential character check - use compiled pattern
        if self._sequential_pattern.search(password):
            result["issues"].append("Password contains sequential characters")
            result["suggestions"].append(
                "Avoid sequential characters like 'abc' or '123'"
            )

        # Repeated character check - use compiled pattern
        if self._repeated_pattern.search(password):
            result["issues"].append("Password contains repeated characters")
            result["suggestions"].append(
                "Avoid repeated characters like 'aaa' or '111'"
            )

        # Calculate final score
        result["score"] = max(0, min(5, result["score"]))

        return result

    def validate_url(self, url: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def validate_ip_address(self, ip: str) -> bool:
        """Validate IP address format."""
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False

    def validate_phone_number(self, phone: str) -> bool:
        """Validate phone number format."""
        # Remove all non-digit characters
        digits = re.sub(r"\D", "", phone)

        # Check if it's a valid length (7-15 digits)
        return 7 <= len(digits) <= 15

    def validate_username(self, username: str) -> dict[str, Any]:
        """Validate username format."""
        result: dict[str, Any] = {"is_valid": True, "issues": [], "suggestions": []}

        if not username or not isinstance(username, str):
            result["is_valid"] = False
            result["issues"].append("Username is required")
            return result

        # Length check
        if len(username) < 3:
            result["is_valid"] = False
            result["issues"].append("Username must be at least 3 characters long")
        elif len(username) > 30:
            result["is_valid"] = False
            result["issues"].append("Username must be no more than 30 characters long")

        # Character check
        if not re.match(r"^[a-zA-Z0-9_-]+$", username):
            result["is_valid"] = False
            result["issues"].append(
                "Username can only contain letters, numbers, underscores, and hyphens"
            )

        # Start/end check
        if username.startswith(("-", "_")):
            result["is_valid"] = False
            result["issues"].append("Username cannot start with a hyphen or underscore")

        if username.endswith(("-", "_")):
            result["is_valid"] = False
            result["issues"].append("Username cannot end with a hyphen or underscore")

        # Reserved usernames
        reserved = [
            "admin",
            "root",
            "administrator",
            "api",
            "www",
            "mail",
            "ftp",
            "support",
        ]
        if username.lower() in reserved:
            result["is_valid"] = False
            result["issues"].append("Username is reserved")

        return result

    def detect_sql_injection(self, value: str) -> bool:
        """Detect potential SQL injection with targeted patterns."""
        if not isinstance(value, str):
            return False  # type: ignore[unreachable]

        # More targeted SQL injection patterns - only flag suspicious combinations
        patterns = [
            # Classic SQL injection techniques
            r"(\b(OR|AND)\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+['\"]?\s*(--|#))",  # Boolean-based with comments
            r"(\bUNION\s+(ALL\s+)?SELECT\s+.*\s+FROM\s+.*\s*(--|#))",  # Union-based with comments
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\s+.*\s+(--|#|\/\*))",  # SQL with comments
            # Time-based blind SQL injection
            r"(\b(WAITFOR\s+DELAY|SLEEP\s*\(|BENCHMARK\s*\()\s*['\"]?\d+['\"]?\s*\)?)",  # Time delays
            r"(\b(IF|CASE)\s*\(.*\s*,\s*SLEEP\s*\(|WAITFOR\s+DELAY)",  # Conditional time delays
            # Error-based SQL injection
            r"(\b(EXTRACTVALUE|UPDATEXML)\s*\(.*,\s*['\"].*['\"].*\)\s*(--|#))",  # Error-based with comments
            r"(\b(FLOOR|RAND)\s*\(.*\s*,\s*.*\s*\)\s*(--|#))",  # Error-based functions
            # Database schema enumeration
            r"(\b(INFORMATION_SCHEMA|SYSOBJECTS|SYSCOLUMNS)\s+.*\s+(--|#))",  # Schema queries with comments
            r"(\b(SELECT\s+.*\s+FROM\s+INFORMATION_SCHEMA|SYSOBJECTS|SYSCOLUMNS)\s+(--|#))",  # Schema enumeration
            # Stacked queries and system commands
            r"(\b(EXEC|EXECUTE|SP_EXECUTESQL)\s+.*\s+(--|#))",  # Stacked queries with comments
            r"(\b(LOAD_FILE|INTO\s+OUTFILE|INTO\s+DUMPFILE)\s+.*\s+(--|#))",  # File operations with comments
            # Privilege escalation attempts
            r"(\b(GRANT|REVOKE)\s+.*\s+(--|#))",  # Privilege escalation with comments
            # Data exfiltration patterns
            r"(\b(SELECT\s+.*\s+INTO\s+OUTFILE|DUMPFILE)\s+.*\s+(--|#))",  # Data exfiltration with comments
            # Suspicious string manipulation in SQL context
            r"(\b(CHAR|ASCII|SUBSTRING|CONCAT)\s*\(.*\s*\)\s*.*\s+(--|#))",  # String functions with comments
            # Database fingerprinting with comments
            r"(\b(USER\(\)|DATABASE\(\)|VERSION\(\))\s*.*\s+(--|#))",  # Database info with comments
        ]

        # Whitelist of legitimate patterns that might trigger false positives
        whitelist_patterns = [
            r"^[a-zA-Z0-9\s\-_.,!?]+$",  # Simple alphanumeric text
            r"^[0-9]+$",  # Pure numbers
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",  # Email addresses
            r"^https?://[^\s]+$",  # URLs
            r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$",  # Dates
            r"^[0-9]{2}:[0-9]{2}:[0-9]{2}$",  # Times
        ]

        # Check whitelist first
        for whitelist_pattern in whitelist_patterns:
            if re.match(whitelist_pattern, value.strip()):
                return False

        # Check for patterns
        for pattern in patterns:
            if re.search(pattern, value, re.IGNORECASE):
                self.logger.warning(f"SQL injection pattern detected: {pattern}")
                return True

        # Check for suspicious character sequences
        suspicious_sequences = [
            "' OR '1'='1",
            "' OR 1=1--",
            "' OR 1=1#",
            "' OR 1=1/*",
            "') OR ('1'='1",
            "') OR (1=1--",
            '" OR "1"="1',
            '" OR 1=1--',
            "1' OR '1'='1",
            "1' OR 1=1--",
            "1') OR ('1'='1",
            "1') OR (1=1--",
            '1" OR "1"="1',
            '1" OR 1=1--',
            "'; DROP TABLE",
            "'; DELETE FROM",
            "'; UPDATE SET",
            "'; INSERT INTO",
            "UNION SELECT",
            "UNION ALL SELECT",
        ]

        value_upper = value.upper()
        for sequence in suspicious_sequences:
            if sequence.upper() in value_upper:
                self.logger.warning(f"SQL injection sequence detected: {sequence}")
                return True

        return False

    def detect_xss(self, value: str) -> bool:
        """Detect potential XSS."""
        if not isinstance(value, str):
            return False  # type: ignore[unreachable]

        # XSS patterns
        patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
            r"<form[^>]*>",
            r"<input[^>]*>",
            r"<link[^>]*>",
            r"<meta[^>]*>",
            r"<style[^>]*>",
            r"expression\s*\(",
            r"url\s*\(",
            r"@import",
        ]

        for pattern in patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True

        return False

    def detect_path_traversal(self, value: str) -> bool:
        """Detect potential path traversal."""
        if not isinstance(value, str):
            return False  # type: ignore[unreachable]

        # Path traversal patterns
        patterns = [
            r"\.\./",
            r"\.\.\\",
            r"\.\.%2f",
            r"\.\.%5c",
            r"\.\.%252f",
            r"\.\.%255c",
            r"\.\.%c0%af",
            r"\.\.%c1%9c",
        ]

        for pattern in patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True

        return False

    def detect_command_injection(self, value: str) -> bool:
        """Detect potential command injection."""
        if not isinstance(value, str):
            return False  # type: ignore[unreachable]

        # Check for command separators
        if self._detect_command_separators(value):
            return True

        # Check for dangerous commands
        if self._detect_dangerous_commands(value):
            return True

        # Check for command chaining
        if self._detect_command_chaining(value):
            return True

        return False

    def _detect_command_separators(self, value: str) -> bool:
        """Detect command separator characters."""
        separators = [r"[;&|`$]", r"&&", r"\|\|", r"\|\|", r"`.*`"]

        for pattern in separators:
            if re.search(pattern, value, re.IGNORECASE):
                self.logger.warning(f"Command separator detected: {pattern}")
                return True
        return False

    def _detect_dangerous_commands(self, value: str) -> bool:
        """Detect dangerous system commands."""
        # Group commands by category for better maintainability
        command_categories = {
            "file_operations": [
                "cat",
                "ls",
                "dir",
                "type",
                "more",
                "less",
                "head",
                "tail",
                "grep",
                "find",
                "awk",
                "sed",
                "cut",
                "sort",
                "uniq",
                "wc",
                "cp",
                "mv",
                "rm",
                "mkdir",
                "rmdir",
                "ln",
                "tar",
                "gzip",
                "gunzip",
                "zip",
                "unzip",
            ],
            "system_info": [
                "ps",
                "top",
                "htop",
                "df",
                "du",
                "free",
                "uptime",
                "whoami",
                "id",
                "groups",
                "w",
                "who",
                "last",
                "lastlog",
                "history",
            ],
            "privilege_escalation": [
                "sudo",
                "su",
                "passwd",
                "chmod",
                "chown",
                "chgrp",
                "umask",
            ],
            "system_control": [
                "kill",
                "killall",
                "pkill",
                "umount",
                "mount",
                "fdisk",
                "mkfs",
                "fsck",
                "dd",
                "systemctl",
                "service",
                "init",
                "chkconfig",
                "update-rc",
                "rc-update",
                "rc-status",
                "rc-service",
                "openrc",
                "systemd",
                "upstart",
                "launchd",
                "daemon",
            ],
            "scheduling": ["cron", "at", "batch", "anacron"],
            "logging": ["logrotate", "syslog", "rsyslog", "journalctl", "dmesg"],
            "network": [
                "wget",
                "curl",
                "ftp",
                "telnet",
                "ssh",
                "scp",
                "rsync",
                "nc",
                "netcat",
                "nmap",
                "ping",
                "traceroute",
                "tracert",
                "nslookup",
                "dig",
                "host",
                "arp",
                "route",
                "iptables",
                "ufw",
                "firewall",
            ],
        }

        for category, commands in command_categories.items():
            for command in commands:
                pattern = rf"\b{re.escape(command)}\b"
                if re.search(pattern, value, re.IGNORECASE):
                    self.logger.warning(
                        f"Dangerous command detected: {command} (category: {category})"
                    )
                    return True
        return False

    def _detect_command_chaining(self, value: str) -> bool:
        """Detect command chaining patterns."""
        chaining_patterns = [
            r";\s*\w+",  # Command after semicolon
            r"&&\s*\w+",  # Command after &&
            r"\|\|\s*\w+",  # Command after ||
            r"\|\s*\w+",  # Command after pipe
            r"`[^`]+`",  # Command substitution
            r"\$\([^)]+\)",  # Command substitution
        ]

        for pattern in chaining_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                self.logger.warning(f"Command chaining detected: {pattern}")
                return True
        return False

    def validate_json(self, json_str: str) -> bool:
        """Validate JSON format."""
        try:
            import json

            json.loads(json_str)
            return True
        except (ValueError, TypeError):
            return False

    def validate_file_extension(
        self, filename: str, allowed_extensions: list[str]
    ) -> bool:
        """Validate file extension."""
        if not filename or not isinstance(filename, str):
            return False

        # Get file extension
        if "." not in filename:
            return False

        extension = filename.split(".")[-1].lower()
        return extension in [ext.lower() for ext in allowed_extensions]

    def validate_file_size(self, file_size: int, max_size: int) -> bool:
        """Validate file size."""
        return 0 <= file_size <= max_size

    def sanitize_input(self, value: str, max_length: int = 1000) -> str:
        """Sanitize input string."""
        if not isinstance(value, str):
            return str(value)  # type: ignore[unreachable]

        # Remove null bytes and control characters
        sanitized = "".join(
            char for char in value if ord(char) >= 32 or char in "\t\n\r"
        )

        # Truncate if too long
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        return sanitized.strip()

    def validate_api_key_format(self, api_key: str) -> bool:
        """Validate API key format."""
        if not api_key or not isinstance(api_key, str):
            return False

        # Check length (should be at least 32 characters)
        if len(api_key) < 32:
            return False

        # Check if it contains only valid characters
        if not re.match(r"^[a-zA-Z0-9_-]+$", api_key):
            return False

        return True

    def validate_jwt_format(self, token: str) -> bool:
        """Validate JWT format."""
        if not token or not isinstance(token, str):
            return False

        # JWT should have 3 parts separated by dots
        parts = token.split(".")
        if len(parts) != 3:
            return False

        # Each part should be base64 encoded
        try:
            import base64

            for part in parts:
                base64.urlsafe_b64decode(part + "==")  # Add padding
        except Exception:
            return False

        return True

    def validate_cors_origin(self, origin: str, allowed_origins: list[str]) -> bool:
        """Validate CORS origin."""
        if not origin or not isinstance(origin, str):
            return False

        # Check exact match
        if origin in allowed_origins:
            return True

        # Check wildcard patterns
        for allowed in allowed_origins:
            if "*" in allowed:
                pattern = allowed.replace("*", ".*")
                if re.match(pattern, origin):
                    return True

        return False

    def validate_content_type(
        self, content_type: str, allowed_types: list[str]
    ) -> bool:
        """Validate content type."""
        if not content_type or not isinstance(content_type, str):
            return False

        # Check if content type starts with any allowed type
        for allowed_type in allowed_types:
            if content_type.startswith(allowed_type):
                return True

        return False

    def validate_input_security(
        self, value: str, input_type: str = "general"
    ) -> dict[str, Any]:
        """
        Comprehensive input security validation.

        Args:
            value: Input value to validate
            input_type: Type of input (email, username, password, general, etc.)

        Returns:
            Dictionary with validation results
        """
        result: dict[str, Any] = {
            "is_valid": True,
            "is_safe": True,
            "threats_detected": [],
            "warnings": [],
            "sanitized_value": value,
        }

        # Length validation
        if len(value) > 10000:  # Reasonable limit
            result["is_valid"] = False
            result["threats_detected"].append("Input too long")
            return result

        # Check for SQL injection
        if self.detect_sql_injection(value):
            result["is_safe"] = False
            result["threats_detected"].append("SQL injection detected")

        # Check for XSS
        if self.detect_xss(value):
            result["is_safe"] = False
            result["threats_detected"].append("XSS detected")

        # Check for command injection
        if self.detect_command_injection(value):
            result["is_safe"] = False
            result["threats_detected"].append("Command injection detected")

        # Check for path traversal
        if self.detect_path_traversal(value):
            result["is_safe"] = False
            result["threats_detected"].append("Path traversal detected")

        # Type-specific validation
        if input_type == "email":
            if not self.validate_email(value):
                result["is_valid"] = False
                result["threats_detected"].append("Invalid email format")
        elif input_type == "username":
            username_result = self.validate_username(value)
            if not username_result["is_valid"]:
                result["is_valid"] = False
                result["threats_detected"].extend(username_result["issues"])
        elif input_type == "password":
            password_result = self.validate_password(value)
            if not password_result["is_valid"]:
                result["is_valid"] = False
                result["threats_detected"].extend(password_result["issues"])
            result["warnings"].extend(password_result["suggestions"])

        # Sanitize input if safe
        if result["is_safe"] and result["is_valid"]:
            result["sanitized_value"] = self.sanitize_input(value)

        # Log security threats
        if result["threats_detected"]:
            self.logger.warning(
                f"Security threats detected in {input_type} input: {result['threats_detected']}"
            )

        return result

    def validate_request_headers(self, headers: dict[str, str]) -> dict[str, Any]:
        """Validate request headers for security."""
        result: dict[str, Any] = {"is_valid": True, "issues": [], "warnings": []}

        # Check for suspicious headers
        suspicious_headers = [
            "x-forwarded-for",
            "x-real-ip",
            "x-originating-ip",
            "x-remote-ip",
            "x-remote-addr",
            "x-client-ip",
        ]

        for header in suspicious_headers:
            if header in headers:
                result["warnings"].append(f"Suspicious header detected: {header}")

        # Check for missing security headers
        security_headers = [
            "user-agent",
            "accept",
            "accept-language",
            "accept-encoding",
        ]

        for header in security_headers:
            if header not in headers:
                result["warnings"].append(f"Missing recommended header: {header}")

        return result


# Global validator instance
_validator: SecurityValidator | None = None


def get_validator() -> SecurityValidator:
    """Get global validator instance."""
    global _validator
    if _validator is None:
        _validator = SecurityValidator()
    return _validator


def setup_validation() -> SecurityValidator:
    """Setup global validator."""
    global _validator
    _validator = SecurityValidator()
    return _validator
