"""Authorization and access control implementation."""

from typing import Any
from enum import Enum
from dataclasses import dataclass

from ..observability.logging import get_logger

logger = get_logger(__name__)


class Permission(str, Enum):
    """System permissions."""

    # User permissions
    USERS_READ = "users:read"
    USERS_WRITE = "users:write"
    USERS_DELETE = "users:delete"

    # Agent permissions
    AGENTS_READ = "agents:read"
    AGENTS_WRITE = "agents:write"
    AGENTS_DELETE = "agents:delete"
    AGENTS_EXECUTE = "agents:execute"

    # Session permissions
    SESSIONS_READ = "sessions:read"
    SESSIONS_WRITE = "sessions:write"
    SESSIONS_DELETE = "sessions:delete"

    # Message permissions
    MESSAGES_READ = "messages:read"
    MESSAGES_WRITE = "messages:write"
    MESSAGES_DELETE = "messages:delete"

    # Tool permissions
    TOOLS_READ = "tools:read"
    TOOLS_WRITE = "tools:write"
    TOOLS_DELETE = "tools:delete"
    TOOLS_EXECUTE = "tools:execute"

    # API Key permissions
    API_KEYS_READ = "api_keys:read"
    API_KEYS_WRITE = "api_keys:write"
    API_KEYS_DELETE = "api_keys:delete"

    # System permissions
    SYSTEM_READ = "system:read"
    SYSTEM_WRITE = "system:write"
    SYSTEM_ADMIN = "system:admin"


class ResourceType(str, Enum):
    """Resource types for authorization."""

    USER = "user"
    AGENT = "agent"
    SESSION = "session"
    MESSAGE = "message"
    TOOL = "tool"
    API_KEY = "api_key"
    SYSTEM = "system"


@dataclass
class Resource:
    """Resource for authorization."""

    type: ResourceType
    id: str
    owner_id: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Policy:
    """Authorization policy."""

    name: str
    description: str
    permissions: list[Permission]
    resources: list[ResourceType]
    conditions: dict[str, Any] | None = None


class AuthorizationError(Exception):
    """Authorization error."""

    pass


class InsufficientPermissionsError(AuthorizationError):
    """Insufficient permissions error."""

    pass


class ResourceNotFoundError(AuthorizationError):
    """Resource not found error."""

    pass


class RBACManager:
    """Role-Based Access Control manager."""

    def __init__(self) -> None:
        self.roles: dict[str, set[Permission]] = {}
        self.policies: dict[str, Policy] = {}
        self.logger = get_logger(__name__)
        self._setup_default_roles()

    def _setup_default_roles(self) -> None:
        """Setup default roles and permissions."""
        # Admin role - full access
        self.roles["admin"] = {
            Permission.USERS_READ,
            Permission.USERS_WRITE,
            Permission.USERS_DELETE,
            Permission.AGENTS_READ,
            Permission.AGENTS_WRITE,
            Permission.AGENTS_DELETE,
            Permission.AGENTS_EXECUTE,
            Permission.SESSIONS_READ,
            Permission.SESSIONS_WRITE,
            Permission.SESSIONS_DELETE,
            Permission.MESSAGES_READ,
            Permission.MESSAGES_WRITE,
            Permission.MESSAGES_DELETE,
            Permission.TOOLS_READ,
            Permission.TOOLS_WRITE,
            Permission.TOOLS_DELETE,
            Permission.TOOLS_EXECUTE,
            Permission.API_KEYS_READ,
            Permission.API_KEYS_WRITE,
            Permission.API_KEYS_DELETE,
            Permission.SYSTEM_READ,
            Permission.SYSTEM_WRITE,
            Permission.SYSTEM_ADMIN,
        }

        # User role - standard user access
        self.roles["user"] = {
            Permission.AGENTS_READ,
            Permission.AGENTS_WRITE,
            Permission.AGENTS_EXECUTE,
            Permission.SESSIONS_READ,
            Permission.SESSIONS_WRITE,
            Permission.MESSAGES_READ,
            Permission.MESSAGES_WRITE,
            Permission.TOOLS_READ,
            Permission.TOOLS_EXECUTE,
            Permission.API_KEYS_READ,
            Permission.API_KEYS_WRITE,
        }

        # Readonly role - read-only access
        self.roles["readonly"] = {
            Permission.AGENTS_READ,
            Permission.SESSIONS_READ,
            Permission.MESSAGES_READ,
            Permission.TOOLS_READ,
        }

        # Service role - API access
        self.roles["service"] = {
            Permission.AGENTS_READ,
            Permission.AGENTS_WRITE,
            Permission.AGENTS_EXECUTE,
            Permission.SESSIONS_READ,
            Permission.SESSIONS_WRITE,
            Permission.MESSAGES_READ,
            Permission.MESSAGES_WRITE,
            Permission.TOOLS_READ,
            Permission.TOOLS_EXECUTE,
        }

    def add_role(self, name: str, permissions: set[Permission]) -> None:
        """Add a new role."""
        self.roles[name] = permissions
        self.logger.info(f"Added role: {name} with {len(permissions)} permissions")

    def remove_role(self, name: str) -> bool:
        """Remove a role."""
        if name in self.roles:
            del self.roles[name]
            self.logger.info(f"Removed role: {name}")
            return True
        return False

    def get_role_permissions(self, role: str) -> set[Permission]:
        """Get permissions for a role."""
        return self.roles.get(role, set())

    def add_policy(self, policy: Policy) -> None:
        """Add an authorization policy."""
        self.policies[policy.name] = policy
        self.logger.info(f"Added policy: {policy.name}")

    def remove_policy(self, name: str) -> bool:
        """Remove a policy."""
        if name in self.policies:
            del self.policies[name]
            self.logger.info(f"Removed policy: {name}")
            return True
        return False

    def get_policy(self, name: str) -> Policy | None:
        """Get a policy by name."""
        return self.policies.get(name)

    def list_roles(self) -> list[str]:
        """List all roles."""
        return list(self.roles.keys())

    def list_policies(self) -> list[str]:
        """List all policies."""
        return list(self.policies.keys())


class AuthorizationService:
    """Main authorization service."""

    def __init__(self, rbac_manager: RBACManager | None = None):
        self.rbac_manager = rbac_manager or RBACManager()
        self.logger = get_logger(__name__)

    def check_permission(
        self, user_permissions: set[Permission], required_permission: Permission
    ) -> bool:
        """Check if user has required permission."""
        return required_permission in user_permissions

    def check_permissions(
        self,
        user_permissions: set[Permission],
        required_permissions: list[Permission],
        require_all: bool = True,
    ) -> bool:
        """Check if user has required permissions."""
        if require_all:
            return all(perm in user_permissions for perm in required_permissions)
        else:
            return any(perm in user_permissions for perm in required_permissions)

    def check_resource_access(
        self, user_permissions: set[Permission], resource: Resource, action: Permission
    ) -> bool:
        """Check if user can perform action on resource."""
        # Check basic permission
        if not self.check_permission(user_permissions, action):
            return False

        # Check resource type permission
        resource_permission = Permission(
            f"{resource.type.value}:{action.value.split(':')[1]}"
        )
        if not self.check_permission(user_permissions, resource_permission):
            return False

        # Check ownership (if applicable)
        if resource.owner_id and action in [
            Permission.AGENTS_WRITE,
            Permission.SESSIONS_WRITE,
            Permission.MESSAGES_WRITE,
        ]:
            # This would need user context to check ownership
            # For now, assume ownership check is handled elsewhere
            pass

        return True

    def filter_resources_by_permission(
        self,
        user_permissions: set[Permission],
        resources: list[Resource],
        action: Permission,
    ) -> list[Resource]:
        """Filter resources based on user permissions."""
        accessible_resources = []

        for resource in resources:
            if self.check_resource_access(user_permissions, resource, action):
                accessible_resources.append(resource)

        return accessible_resources

    def get_user_permissions_for_roles(self, roles: list[str]) -> set[Permission]:
        """Get all permissions for a list of roles."""
        permissions = set()

        for role in roles:
            role_permissions = self.rbac_manager.get_role_permissions(role)
            permissions.update(role_permissions)

        return permissions

    def can_access_resource(
        self,
        user_id: str,
        user_roles: list[str],
        resource: Resource,
        action: Permission,
    ) -> bool:
        """Check if user can access resource."""
        user_permissions = self.get_user_permissions_for_roles(user_roles)
        return self.check_resource_access(user_permissions, resource, action)

    def get_accessible_resources(
        self,
        user_id: str,
        user_roles: list[str],
        resources: list[Resource],
        action: Permission,
    ) -> list[Resource]:
        """Get resources accessible to user."""
        user_permissions = self.get_user_permissions_for_roles(user_roles)
        return self.filter_resources_by_permission(user_permissions, resources, action)

    def create_resource_policy(
        self, resource_type: ResourceType, owner_id: str, permissions: list[Permission]
    ) -> Policy:
        """Create a resource-specific policy."""
        policy_name = f"{resource_type.value}_{owner_id}_policy"

        policy = Policy(
            name=policy_name,
            description=f"Policy for {resource_type.value} owned by {owner_id}",
            permissions=permissions,
            resources=[resource_type],
            conditions={"owner_id": owner_id},
        )

        self.rbac_manager.add_policy(policy)
        return policy

    def apply_policy(
        self,
        user_permissions: set[Permission],
        policy: Policy,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Apply a policy to user permissions."""
        # Check if user has required permissions for policy
        if not self.check_permissions(
            user_permissions, policy.permissions, require_all=True
        ):
            return False

        # Check conditions if any
        if policy.conditions and context:
            for key, value in policy.conditions.items():
                if context.get(key) != value:
                    return False

        return True

    def evaluate_access(
        self,
        user_id: str,
        user_roles: list[str],
        resource: Resource,
        action: Permission,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Evaluate access with all policies and conditions."""
        user_permissions = self.get_user_permissions_for_roles(user_roles)

        # Check basic resource access
        if not self.check_resource_access(user_permissions, resource, action):
            return False

        # Apply relevant policies
        for policy in self.rbac_manager.policies.values():
            if resource.type in policy.resources:
                if not self.apply_policy(user_permissions, policy, context):
                    return False

        return True

    def get_effective_permissions(
        self, user_roles: list[str], resource_type: ResourceType | None = None
    ) -> set[Permission]:
        """Get effective permissions for user roles."""
        permissions = self.get_user_permissions_for_roles(user_roles)

        if resource_type:
            # Filter permissions for specific resource type
            filtered_permissions = set()
            for perm in permissions:
                if perm.value.startswith(f"{resource_type.value}:"):
                    filtered_permissions.add(perm)
            return filtered_permissions

        return permissions

    def create_custom_role(
        self, name: str, permissions: list[Permission], description: str = ""
    ) -> None:
        """Create a custom role."""
        self.rbac_manager.add_role(name, set(permissions))
        self.logger.info(
            f"Created custom role: {name} with {len(permissions)} permissions"
        )

    def update_role_permissions(self, role: str, permissions: list[Permission]) -> bool:
        """Update role permissions."""
        if role not in self.rbac_manager.roles:
            return False

        self.rbac_manager.roles[role] = set(permissions)
        self.logger.info(f"Updated role {role} with {len(permissions)} permissions")
        return True

    def get_role_hierarchy(self) -> dict[str, list[str]]:
        """Get role hierarchy (which roles inherit from which)."""
        # This would be implemented based on your role hierarchy requirements
        return {
            "admin": ["user", "readonly"],
            "user": ["readonly"],
            "service": ["readonly"],
        }

    def check_role_inheritance(self, user_roles: list[str], required_role: str) -> bool:
        """Check if user has required role through inheritance."""
        hierarchy = self.get_role_hierarchy()

        for role in user_roles:
            if role == required_role:
                return True

            # Check inheritance
            inherited_roles = hierarchy.get(role, [])
            if required_role in inherited_roles:
                return True

        return False


# Global authorization service instance
_authz_service: AuthorizationService | None = None


def get_authz_service() -> AuthorizationService:
    """Get global authorization service instance."""
    global _authz_service
    if _authz_service is None:
        _authz_service = AuthorizationService()
    return _authz_service


def setup_authorization(
    rbac_manager: RBACManager | None = None,
) -> AuthorizationService:
    """Setup global authorization service."""
    global _authz_service
    _authz_service = AuthorizationService(rbac_manager)
    return _authz_service
