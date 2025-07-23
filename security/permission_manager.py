"""
Permission Manager for ChatBI platform.
Handles user permissions, access control, and resource authorization.
"""

import yaml
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import time

from utils.logger import get_logger
from utils.exceptions import AuthorizationException, ErrorCodes
from .audit_logger import AuditLogger

logger = get_logger(__name__)


class PermissionLevel(Enum):
    """Permission levels for different operations."""
    NONE = 0
    READ = 1
    WRITE = 2
    ADMIN = 3


class ResourceType(Enum):
    """Types of resources that can be protected."""
    TABLE = "table"
    COLUMN = "column"
    DATABASE = "database"
    FUNCTION = "function"
    VIEW = "view"


class PermissionManager:
    """
    Comprehensive permission management system for controlling access
    to data sources, tables, columns, and operations.
    """

    def __init__(self, permissions_file: str = "data/permissions.yaml"):
        """Initialize permission manager with configuration."""
        self.permissions_file = permissions_file
        self.audit_logger = AuditLogger()

        # In-memory permission cache for performance
        self.permission_cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps = {}

        # Default permissions
        self.default_permissions = {
            "tables": {},
            "columns": {},
            "operations": {
                "SELECT": True,
                "INSERT": False,
                "UPDATE": False,
                "DELETE": False,
                "CREATE": False,
                "DROP": False
            }
        }

        # Load permissions from file
        self.permissions_config = self._load_permissions_config()

        # Role-based access control
        self.roles = {
            "admin": {
                "permissions": ["*"],
                "restrictions": []
            },
            "analyst": {
                "permissions": ["SELECT", "VIEW"],
                "restrictions": ["system_tables"]
            },
            "viewer": {
                "permissions": ["SELECT"],
                "restrictions": ["sensitive_columns", "system_tables"]
            },
            "guest": {
                "permissions": ["SELECT"],
                "restrictions": ["all_tables_except_public"]
            }
        }

    async def check_table_access(
            self,
            user_id: int,
            table_name: str,
            operation: str = "SELECT"
    ) -> bool:
        """
        Check if user has permission to access a specific table.

        Args:
            user_id: ID of the user
            table_name: Name of the table
            operation: Operation to perform (SELECT, INSERT, etc.)

        Returns:
            True if access is allowed, False otherwise
        """
        try:
            # Check cache first
            cache_key = f"table_{user_id}_{table_name}_{operation}"
            cached_result = self._get_cached_permission(cache_key)
            if cached_result is not None:
                return cached_result

            # Get user permissions
            user_permissions = await self._get_user_permissions(user_id)

            # Check global operation permission
            if not user_permissions.get("operations", {}).get(operation, False):
                await self._audit_access_denied(
                    user_id, ResourceType.TABLE, table_name, operation, "operation_not_allowed"
                )
                self._cache_permission(cache_key, False)
                return False

            # Check table-specific permissions
            table_permissions = user_permissions.get("tables", {})

            # Check if table is explicitly allowed
            if table_name in table_permissions:
                allowed = table_permissions[table_name].get(operation, False)
                self._cache_permission(cache_key, allowed)

                if not allowed:
                    await self._audit_access_denied(
                        user_id, ResourceType.TABLE, table_name, operation, "table_access_denied"
                    )

                return allowed

            # Check wildcard permissions
            for pattern, permissions in table_permissions.items():
                if self._match_pattern(table_name, pattern):
                    allowed = permissions.get(operation, False)
                    self._cache_permission(cache_key, allowed)

                    if not allowed:
                        await self._audit_access_denied(
                            user_id, ResourceType.TABLE, table_name, operation, "pattern_access_denied"
                        )

                    return allowed

            # Check role-based permissions
            user_roles = await self._get_user_roles(user_id)
            for role in user_roles:
                if self._check_role_table_access(role, table_name, operation):
                    self._cache_permission(cache_key, True)
                    return True

            # Default deny
            await self._audit_access_denied(
                user_id, ResourceType.TABLE, table_name, operation, "default_deny"
            )
            self._cache_permission(cache_key, False)
            return False

        except Exception as e:
            logger.error(f"Permission check failed for user {user_id}, table {table_name}: {e}")
            # Fail secure - deny access on error
            return False

    async def check_column_access(
            self,
            user_id: int,
            table_name: str,
            column_name: str,
            operation: str = "SELECT"
    ) -> bool:
        """
        Check if user has permission to access a specific column.

        Args:
            user_id: ID of the user
            table_name: Name of the table
            column_name: Name of the column
            operation: Operation to perform

        Returns:
            True if access is allowed, False otherwise
        """
        try:
            # First check table access
            if not await self.check_table_access(user_id, table_name, operation):
                return False

            # Check cache
            cache_key = f"column_{user_id}_{table_name}_{column_name}_{operation}"
            cached_result = self._get_cached_permission(cache_key)
            if cached_result is not None:
                return cached_result

            # Get user permissions
            user_permissions = await self._get_user_permissions(user_id)

            # Check column-specific permissions
            column_permissions = user_permissions.get("columns", {})
            full_column_name = f"{table_name}.{column_name}"

            # Check explicit column permission
            if full_column_name in column_permissions:
                allowed = column_permissions[full_column_name].get(operation, True)
                self._cache_permission(cache_key, allowed)

                if not allowed:
                    await self._audit_access_denied(
                        user_id, ResourceType.COLUMN, full_column_name, operation, "column_access_denied"
                    )

                return allowed

            # Check column patterns
            for pattern, permissions in column_permissions.items():
                if self._match_pattern(full_column_name, pattern):
                    allowed = permissions.get(operation, True)
                    self._cache_permission(cache_key, allowed)

                    if not allowed:
                        await self._audit_access_denied(
                            user_id, ResourceType.COLUMN, full_column_name, operation, "pattern_column_denied"
                        )

                    return allowed

            # Check for sensitive column restrictions
            if self._is_sensitive_column(table_name, column_name):
                user_roles = await self._get_user_roles(user_id)
                if not any(self._role_can_access_sensitive_data(role) for role in user_roles):
                    await self._audit_access_denied(
                        user_id, ResourceType.COLUMN, full_column_name, operation, "sensitive_data_denied"
                    )
                    self._cache_permission(cache_key, False)
                    return False

            # Default allow if table access is granted
            self._cache_permission(cache_key, True)
            return True

        except Exception as e:
            logger.error(f"Column permission check failed for user {user_id}, column {table_name}.{column_name}: {e}")
            return False

    async def check_operation_permission(
            self,
            user_id: int,
            operation: str,
            resource_type: ResourceType = ResourceType.TABLE
    ) -> bool:
        """
        Check if user has permission to perform a specific operation.

        Args:
            user_id: ID of the user
            operation: Operation to check
            resource_type: Type of resource

        Returns:
            True if operation is allowed, False otherwise
        """
        try:
            cache_key = f"operation_{user_id}_{operation}_{resource_type.value}"
            cached_result = self._get_cached_permission(cache_key)
            if cached_result is not None:
                return cached_result

            user_permissions = await self._get_user_permissions(user_id)
            allowed = user_permissions.get("operations", {}).get(operation, False)

            # Check role-based operation permissions
            if not allowed:
                user_roles = await self._get_user_roles(user_id)
                for role in user_roles:
                    if self._check_role_operation_permission(role, operation):
                        allowed = True
                        break

            self._cache_permission(cache_key, allowed)

            if not allowed:
                await self._audit_access_denied(
                    user_id, resource_type, operation, operation, "operation_permission_denied"
                )

            return allowed

        except Exception as e:
            logger.error(f"Operation permission check failed for user {user_id}, operation {operation}: {e}")
            return False

    async def get_accessible_tables(self, user_id: int) -> List[str]:
        """
        Get list of tables that user has access to.

        Args:
            user_id: ID of the user

        Returns:
            List of accessible table names
        """
        try:
            accessible_tables = []
            user_permissions = await self._get_user_permissions(user_id)

            # Get tables from explicit permissions
            table_permissions = user_permissions.get("tables", {})
            for table_pattern, permissions in table_permissions.items():
                if permissions.get("SELECT", False):
                    if "*" in table_pattern:
                        # Handle wildcard patterns - would need actual table list to expand
                        continue
                    else:
                        accessible_tables.append(table_pattern)

            # Add tables from role permissions
            user_roles = await self._get_user_roles(user_id)
            for role in user_roles:
                role_tables = self._get_role_accessible_tables(role)
                accessible_tables.extend(role_tables)

            return list(set(accessible_tables))

        except Exception as e:
            logger.error(f"Failed to get accessible tables for user {user_id}: {e}")
            return []

    async def grant_permission(
            self,
            admin_user_id: int,
            target_user_id: int,
            resource_type: ResourceType,
            resource_name: str,
            operation: str
    ) -> bool:
        """
        Grant permission to a user (admin operation).

        Args:
            admin_user_id: ID of admin user granting permission
            target_user_id: ID of user receiving permission
            resource_type: Type of resource
            resource_name: Name of the resource
            operation: Operation to grant

        Returns:
            True if permission was granted successfully
        """
        try:
            # Check if admin user has permission to grant
            if not await self._check_admin_permission(admin_user_id, "GRANT"):
                raise AuthorizationException(
                    "Insufficient privileges to grant permissions",
                    ErrorCodes.AUTHZ_INSUFFICIENT_PERMISSIONS
                )

            # Grant the permission (implementation would update database/config)
            success = await self._update_user_permission(
                target_user_id, resource_type, resource_name, operation, True
            )

            if success:
                # Clear cache for affected user
                self._clear_user_cache(target_user_id)

                # Audit the permission grant
                await self.audit_logger.log_security_event(
                    user_id=admin_user_id,
                    event_type="permission_granted",
                    details={
                        "target_user": target_user_id,
                        "resource_type": resource_type.value,
                        "resource_name": resource_name,
                        "operation": operation
                    },
                    risk_level="medium"
                )

                logger.info(f"Permission granted by user {admin_user_id} to user {target_user_id} for {resource_name}")

            return success

        except Exception as e:
            logger.error(f"Failed to grant permission: {e}")
            return False

    async def revoke_permission(
            self,
            admin_user_id: int,
            target_user_id: int,
            resource_type: ResourceType,
            resource_name: str,
            operation: str
    ) -> bool:
        """
        Revoke permission from a user (admin operation).

        Args:
            admin_user_id: ID of admin user revoking permission
            target_user_id: ID of user losing permission
            resource_type: Type of resource
            resource_name: Name of the resource
            operation: Operation to revoke

        Returns:
            True if permission was revoked successfully
        """
        try:
            # Check admin permission
            if not await self._check_admin_permission(admin_user_id, "REVOKE"):
                raise AuthorizationException(
                    "Insufficient privileges to revoke permissions",
                    ErrorCodes.AUTHZ_INSUFFICIENT_PERMISSIONS
                )

            # Revoke the permission
            success = await self._update_user_permission(
                target_user_id, resource_type, resource_name, operation, False
            )

            if success:
                # Clear cache
                self._clear_user_cache(target_user_id)

                # Audit the permission revocation
                await self.audit_logger.log_security_event(
                    user_id=admin_user_id,
                    event_type="permission_revoked",
                    details={
                        "target_user": target_user_id,
                        "resource_type": resource_type.value,
                        "resource_name": resource_name,
                        "operation": operation
                    },
                    risk_level="high"
                )

                logger.info(
                    f"Permission revoked by user {admin_user_id} from user {target_user_id} for {resource_name}")

            return success

        except Exception as e:
            logger.error(f"Failed to revoke permission: {e}")
            return False

    # Private helper methods

    def _load_permissions_config(self) -> Dict[str, Any]:
        """Load permissions configuration from YAML file."""
        try:
            with open(self.permissions_file, 'r') as file:
                config = yaml.safe_load(file) or {}
            logger.info(f"Loaded permissions configuration from {self.permissions_file}")
            return config
        except FileNotFoundError:
            logger.warning(f"Permissions file {self.permissions_file} not found, using defaults")
            return {"users": {}, "roles": {}}
        except Exception as e:
            logger.error(f"Failed to load permissions config: {e}")
            return {"users": {}, "roles": {}}

    async def _get_user_permissions(self, user_id: int) -> Dict[str, Any]:
        """Get permissions for a specific user."""
        # This would typically query the database
        # For now, return default permissions
        user_config = self.permissions_config.get("users", {}).get(str(user_id), {})

        # Merge with defaults
        permissions = self.default_permissions.copy()
        permissions.update(user_config)

        return permissions

    async def _get_user_roles(self, user_id: int) -> List[str]:
        """Get roles assigned to a user."""
        # This would typically query the database
        user_config = self.permissions_config.get("users", {}).get(str(user_id), {})
        return user_config.get("roles", ["viewer"])  # Default role

    def _get_cached_permission(self, cache_key: str) -> Optional[bool]:
        """Get permission from cache if not expired."""
        if cache_key in self.permission_cache:
            timestamp = self.cache_timestamps.get(cache_key, 0)
            if time.time() - timestamp < self.cache_ttl:
                return self.permission_cache[cache_key]
            else:
                # Remove expired entry
                del self.permission_cache[cache_key]
                del self.cache_timestamps[cache_key]
        return None

    def _cache_permission(self, cache_key: str, result: bool):
        """Cache permission result."""
        import time
        self.permission_cache[cache_key] = result
        self.cache_timestamps[cache_key] = time.time()

    def _clear_user_cache(self, user_id: int):
        """Clear all cached permissions for a user."""
        keys_to_remove = [
            key for key in self.permission_cache.keys()
            if key.startswith(f"table_{user_id}_") or
               key.startswith(f"column_{user_id}_") or
               key.startswith(f"operation_{user_id}_")
        ]

        for key in keys_to_remove:
            del self.permission_cache[key]
            if key in self.cache_timestamps:
                del self.cache_timestamps[key]

    def _match_pattern(self, name: str, pattern: str) -> bool:
        """Check if name matches a wildcard pattern."""
        import fnmatch
        return fnmatch.fnmatch(name.lower(), pattern.lower())

    def _check_role_table_access(self, role: str, table_name: str, operation: str) -> bool:
        """Check if role allows access to table."""
        role_config = self.roles.get(role, {})
        permissions = role_config.get("permissions", [])
        restrictions = role_config.get("restrictions", [])

        # Check if operation is allowed
        if "*" not in permissions and operation not in permissions:
            return False

        # Check restrictions
        if "system_tables" in restrictions and self._is_system_table(table_name):
            return False

        if "all_tables_except_public" in restrictions and not table_name.startswith("public_"):
            return False

        return True

    def _check_role_operation_permission(self, role: str, operation: str) -> bool:
        """Check if role allows specific operation."""
        role_config = self.roles.get(role, {})
        permissions = role_config.get("permissions", [])
        return "*" in permissions or operation in permissions

    def _is_sensitive_column(self, table_name: str, column_name: str) -> bool:
        """Check if column contains sensitive data."""
        sensitive_patterns = [
            "password", "ssn", "social_security", "credit_card",
            "phone", "email", "address", "salary", "wage"
        ]

        column_lower = column_name.lower()
        return any(pattern in column_lower for pattern in sensitive_patterns)

    def _is_system_table(self, table_name: str) -> bool:
        """Check if table is a system table."""
        system_prefixes = ["information_schema", "mysql", "performance_schema", "sys"]
        return any(table_name.lower().startswith(prefix) for prefix in system_prefixes)

    def _role_can_access_sensitive_data(self, role: str) -> bool:
        """Check if role can access sensitive data."""
        privileged_roles = ["admin", "analyst"]
        return role in privileged_roles

    def _get_role_accessible_tables(self, role: str) -> List[str]:
        """Get tables accessible to a role."""
        # This would be configured based on role definitions
        if role == "admin":
            return []  # Admin can access all tables
        elif role == "analyst":
            return ["sales", "products", "customers", "orders"]
        elif role == "viewer":
            return ["public_sales", "public_products"]
        else:
            return ["public_demo"]

    async def _check_admin_permission(self, user_id: int, operation: str) -> bool:
        """Check if user has admin permissions."""
        user_roles = await self._get_user_roles(user_id)
        return "admin" in user_roles

    async def _update_user_permission(
            self,
            user_id: int,
            resource_type: ResourceType,
            resource_name: str,
            operation: str,
            grant: bool
    ) -> bool:
        """Update user permission in persistent storage."""
        # This would update the database or configuration file
        # For now, just return True to simulate success
        logger.info(
            f"{'Granted' if grant else 'Revoked'} {operation} permission on {resource_type.value} {resource_name} for user {user_id}")
        return True

    async def _audit_access_denied(
            self,
            user_id: int,
            resource_type: ResourceType,
            resource_name: str,
            operation: str,
            reason: str
    ):
        """Audit access denied events."""
        try:
            await self.audit_logger.log_security_event(
                user_id=user_id,
                event_type="access_denied",
                details={
                    "resource_type": resource_type.value,
                    "resource_name": resource_name,
                    "operation": operation,
                    "reason": reason
                },
                risk_level="medium"
            )
        except Exception as e:
            logger.error(f"Failed to audit access denied event: {e}")

    async def get_permission_summary(self, user_id: int) -> Dict[str, Any]:
        """Get comprehensive permission summary for a user."""
        try:
            user_permissions = await self._get_user_permissions(user_id)
            user_roles = await self._get_user_roles(user_id)
            accessible_tables = await self.get_accessible_tables(user_id)

            return {
                "user_id": user_id,
                "roles": user_roles,
                "operations": user_permissions.get("operations", {}),
                "accessible_tables": accessible_tables,
                "table_count": len(accessible_tables),
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get permission summary for user {user_id}: {e}")
            return {"user_id": user_id, "error": str(e)}