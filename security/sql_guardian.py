"""
SQL Guardian for ChatBI platform.
Advanced SQL security validation and threat detection system.
"""

import re
import hashlib
import time
from typing import Dict, List, Any, Set, Optional
from collections import defaultdict, deque
import sqlparse
from sqlparse.sql import Statement, Token
from sqlparse.tokens import Keyword, String, Comment

from utils.logger import get_logger
from utils.exceptions import SQLSecurityException, ErrorCodes
from .permission_manager import PermissionManager
from .audit_logger import AuditLogger

logger = get_logger(__name__)


class SQLGuardian:
    """
    Advanced SQL security system that provides multi-layered protection
    against SQL injection, unauthorized access, and malicious queries.
    """

    def __init__(self):
        """Initialize SQL Guardian with security rules and monitoring."""
        self.permission_manager = PermissionManager()
        self.audit_logger = AuditLogger()

        # Rate limiting for query validation
        self.validation_cache = {}  # Query hash -> validation result
        self.rate_limiter = defaultdict(lambda: deque(maxlen=100))  # User -> recent queries

        # Security rules configuration
        self.security_rules = {
            "max_query_length": 10000,
            "max_validation_time": 5.0,  # seconds
            "rate_limit_window": 300,  # 5 minutes
            "max_queries_per_window": 50,
            "suspicious_pattern_threshold": 3
        }

        # Advanced threat patterns
        self.threat_patterns = {
            "sql_injection": [
                r"(\bunion\s+all\s+select)",
                r"(\bor\s+1\s*=\s*1)",
                r"(\band\s+1\s*=\s*2)",
                r"(\';\s*drop\s+table)",
                r"(exec\s*\(\s*@)",
                r"(char\s*\(\s*\d+\s*\))",
                r"(waitfor\s+delay)",
                r"(benchmark\s*\()",
                r"(sleep\s*\()",
                r"(load_file\s*\()",
                r"(into\s+outfile)",
                r"(--\s*[^\r\n]*)",
                r"(/\*.*?\*/)",
                r"(xp_cmdshell)",
                r"(sp_executesql)"
            ],
            "data_exfiltration": [
                r"(select\s+\*\s+from\s+information_schema)",
                r"(show\s+tables)",
                r"(show\s+databases)",
                r"(describe\s+\w+)",
                r"(explain\s+select)",
                r"(select.*user\(\))",
                r"(select.*database\(\))",
                r"(select.*version\(\))"
            ],
            "privilege_escalation": [
                r"(grant\s+\w+)",
                r"(revoke\s+\w+)",
                r"(alter\s+user)",
                r"(create\s+user)",
                r"(drop\s+user)",
                r"(set\s+password)",
                r"(flush\s+privileges)"
            ],
            "resource_abuse": [
                r"(select.*from.*cross\s+join)",
                r"(select.*limit\s+\d{6,})",  # Very large limits
                r"(select.*order\s+by.*rand\(\))",
                r"(recursive\s+cte)",
                r"(while\s+\d+\s*=\s*\d+)",  # Infinite loops
                r"(cartesian\s+product)"
            ]
        }

        # Whitelist of safe functions and keywords
        self.safe_functions = {
            'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'ROUND', 'CEIL', 'FLOOR',
            'UPPER', 'LOWER', 'TRIM', 'SUBSTRING', 'LENGTH', 'CONCAT',
            'DATE', 'YEAR', 'MONTH', 'DAY', 'NOW', 'CURDATE', 'DATEDIFF',
            'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'COALESCE', 'NULLIF'
        }

        # Blacklist of dangerous operations
        self.dangerous_keywords = {
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'CREATE', 'ALTER', 'TRUNCATE',
            'GRANT', 'REVOKE', 'EXEC', 'EXECUTE', 'CALL', 'LOAD', 'OUTFILE',
            'INFILE', 'DUMPFILE', 'BACKUP', 'RESTORE', 'SHUTDOWN', 'KILL'
        }

    async def validate_query(
            self,
            sql_query: str,
            user_id: int,
            context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive security validation of SQL query.

        Args:
            sql_query: SQL query to validate
            user_id: ID of user submitting the query
            context: Additional context for validation

        Returns:
            Validation result with security assessment
        """
        start_time = time.time()

        logger.info(f"Validating SQL query for user {user_id}")

        try:
            # Quick cache check for identical queries
            query_hash = self._hash_query(sql_query)
            if query_hash in self.validation_cache:
                cached_result = self.validation_cache[query_hash].copy()
                cached_result["from_cache"] = True
                return cached_result

            # Rate limiting check
            rate_limit_result = self._check_rate_limits(user_id, sql_query)
            if not rate_limit_result["allowed"]:
                raise SQLSecurityException(
                    f"Rate limit exceeded: {rate_limit_result['reason']}",
                    ErrorCodes.SERVICE_RATE_LIMITED
                )

            # Initialize validation result
            validation_result = {
                "is_safe": True,
                "risk_level": "low",
                "security_score": 100,
                "threats_detected": [],
                "warnings": [],
                "blocked_patterns": [],
                "permission_issues": [],
                "metadata": {
                    "query_hash": query_hash,
                    "validation_time": 0,
                    "user_id": user_id
                }
            }

            # Step 1: Basic security checks
            basic_security = await self._perform_basic_security_checks(sql_query)
            self._merge_validation_results(validation_result, basic_security)

            # Step 2: Advanced threat detection
            threat_analysis = await self._perform_threat_analysis(sql_query)
            self._merge_validation_results(validation_result, threat_analysis)

            # Step 3: Permission validation
            permission_check = await self._validate_permissions(sql_query, user_id, context)
            self._merge_validation_results(validation_result, permission_check)

            # Step 4: Query structure analysis
            structure_analysis = await self._analyze_query_structure(sql_query)
            self._merge_validation_results(validation_result, structure_analysis)

            # Step 5: Final risk assessment
            final_assessment = self._calculate_final_risk(validation_result)
            validation_result.update(final_assessment)

            # Cache successful validations
            if validation_result["is_safe"]:
                self.validation_cache[query_hash] = validation_result.copy()

            # Record validation time
            validation_result["metadata"]["validation_time"] = time.time() - start_time

            # Audit log the validation
            await self._audit_validation(user_id, sql_query, validation_result)

            logger.info(f"SQL validation completed: {validation_result['risk_level']} risk")
            return validation_result

        except SQLSecurityException:
            raise
        except Exception as e:
            logger.error(f"SQL validation failed: {e}")
            # Fail secure - block query if validation fails
            return {
                "is_safe": False,
                "risk_level": "critical",
                "security_score": 0,
                "threats_detected": ["validation_failure"],
                "error": f"Validation system error: {str(e)}",
                "metadata": {
                    "query_hash": self._hash_query(sql_query),
                    "validation_time": time.time() - start_time,
                    "user_id": user_id
                }
            }

    async def _perform_basic_security_checks(self, sql_query: str) -> Dict[str, Any]:
        """Perform basic security validation checks."""
        result = {
            "threats_detected": [],
            "warnings": [],
            "security_score": 100
        }

        query_upper = sql_query.upper()
        query_lower = sql_query.lower()

        # Check query length
        if len(sql_query) > self.security_rules["max_query_length"]:
            result["threats_detected"].append("excessive_query_length")
            result["security_score"] -= 30

        # Check for dangerous keywords
        for keyword in self.dangerous_keywords:
            if re.search(rf'\b{keyword}\b', query_upper):
                result["threats_detected"].append(f"dangerous_keyword_{keyword.lower()}")
                result["security_score"] -= 50

        # Check for multiple statements
        if ';' in sql_query.rstrip(';'):
            result["threats_detected"].append("multiple_statements")
            result["security_score"] -= 40

        # Check for SQL comments in suspicious locations
        if '--' in sql_query and not sql_query.strip().startswith('--'):
            result["warnings"].append("inline_comments_detected")
            result["security_score"] -= 10

        # Check for quote escaping
        if "\\'" in sql_query or '\\"' in sql_query:
            result["threats_detected"].append("quote_escaping")
            result["security_score"] -= 25

        # Ensure query starts with SELECT
        first_word = sql_query.strip().split()[0].upper() if sql_query.strip() else ""
        if first_word != "SELECT":
            result["threats_detected"].append("non_select_statement")
            result["security_score"] -= 60

        return result

    async def _perform_threat_analysis(self, sql_query: str) -> Dict[str, Any]:
        """Perform advanced threat pattern analysis."""
        result = {
            "threats_detected": [],
            "blocked_patterns": [],
            "security_score": 100
        }

        query_normalized = sql_query.lower().replace('\n', ' ').replace('\t', ' ')

        # Check each threat category
        for threat_type, patterns in self.threat_patterns.items():
            matches = []
            for pattern in patterns:
                if re.search(pattern, query_normalized, re.IGNORECASE):
                    matches.append(pattern)

            if matches:
                result["threats_detected"].append(threat_type)
                result["blocked_patterns"].extend(matches)

                # Scoring based on threat severity
                if threat_type == "sql_injection":
                    result["security_score"] -= 80
                elif threat_type == "privilege_escalation":
                    result["security_score"] -= 90
                elif threat_type == "data_exfiltration":
                    result["security_score"] -= 60
                elif threat_type == "resource_abuse":
                    result["security_score"] -= 40

        return result

    async def _validate_permissions(
            self,
            sql_query: str,
            user_id: int,
            context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate user permissions for query execution."""
        result = {
            "permission_issues": [],
            "security_score": 100
        }

        try:
            # Extract table names from query
            table_names = self._extract_table_names(sql_query)

            # Check permissions for each table
            for table_name in table_names:
                has_permission = await self.permission_manager.check_table_access(
                    user_id=user_id,
                    table_name=table_name,
                    operation="SELECT"
                )

                if not has_permission:
                    result["permission_issues"].append(f"access_denied_{table_name}")
                    result["security_score"] -= 50

            # Check column-level permissions if specified
            column_names = self._extract_column_names(sql_query)
            for table_name, columns in column_names.items():
                for column in columns:
                    has_permission = await self.permission_manager.check_column_access(
                        user_id=user_id,
                        table_name=table_name,
                        column_name=column,
                        operation="SELECT"
                    )

                    if not has_permission:
                        result["permission_issues"].append(f"column_access_denied_{table_name}.{column}")
                        result["security_score"] -= 20

        except Exception as e:
            logger.warning(f"Permission validation failed: {e}")
            result["permission_issues"].append("permission_check_failed")
            result["security_score"] -= 30

        return result

    async def _analyze_query_structure(self, sql_query: str) -> Dict[str, Any]:
        """Analyze query structure for security issues."""
        result = {
            "warnings": [],
            "security_score": 100
        }

        try:
            parsed = sqlparse.parse(sql_query)[0]

            # Check for deeply nested subqueries
            subquery_depth = self._count_subquery_depth(parsed)
            if subquery_depth > 5:
                result["warnings"].append("deep_nesting_detected")
                result["security_score"] -= 15

            # Check for excessive JOINs
            join_count = self._count_joins(sql_query)
            if join_count > 10:
                result["warnings"].append("excessive_joins")
                result["security_score"] -= 20

            # Check for SELECT *
            if "SELECT *" in sql_query.upper():
                result["warnings"].append("select_star_usage")
                result["security_score"] -= 5

            # Check for LIMIT clause
            if "LIMIT" not in sql_query.upper():
                result["warnings"].append("missing_limit_clause")
                result["security_score"] -= 10

        except Exception as e:
            logger.warning(f"Query structure analysis failed: {e}")
            result["warnings"].append("structure_analysis_failed")

        return result

    def _check_rate_limits(self, user_id: int, sql_query: str) -> Dict[str, Any]:
        """Check if user is within rate limits."""
        current_time = time.time()
        user_queries = self.rate_limiter[user_id]

        # Clean old queries outside the window
        window_start = current_time - self.security_rules["rate_limit_window"]
        while user_queries and user_queries[0] < window_start:
            user_queries.popleft()

        # Check rate limit
        if len(user_queries) >= self.security_rules["max_queries_per_window"]:
            return {
                "allowed": False,
                "reason": f"Rate limit exceeded: {len(user_queries)} queries in {self.security_rules['rate_limit_window']} seconds"
            }

        # Add current query
        user_queries.append(current_time)

        return {"allowed": True}

    def _calculate_final_risk(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate final risk level based on all checks."""
        security_score = validation_result.get("security_score", 100)
        threats = validation_result.get("threats_detected", [])
        permission_issues = validation_result.get("permission_issues", [])

        # Determine risk level
        if security_score >= 80 and not threats and not permission_issues:
            risk_level = "low"
            is_safe = True
        elif security_score >= 60 and len(threats) <= 1:
            risk_level = "medium"
            is_safe = True
        elif security_score >= 40:
            risk_level = "high"
            is_safe = False
        else:
            risk_level = "critical"
            is_safe = False

        # Override if critical threats detected
        critical_threats = ["sql_injection", "privilege_escalation", "dangerous_keyword_drop"]
        if any(threat in threats for threat in critical_threats):
            risk_level = "critical"
            is_safe = False

        return {
            "is_safe": is_safe,
            "risk_level": risk_level,
            "final_score": max(0, min(100, security_score))
        }

    def _merge_validation_results(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Merge validation results from different checks."""
        for key in ["threats_detected", "warnings", "blocked_patterns", "permission_issues"]:
            if key in source:
                target.setdefault(key, []).extend(source[key])

        if "security_score" in source:
            target["security_score"] = min(target.get("security_score", 100), source["security_score"])

    def _hash_query(self, sql_query: str) -> str:
        """Create hash of normalized query for caching."""
        normalized = re.sub(r'\s+', ' ', sql_query.strip().lower())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _extract_table_names(self, sql_query: str) -> Set[str]:
        """Extract table names from SQL query."""
        tables = set()

        try:
            parsed = sqlparse.parse(sql_query)[0]

            def extract_tables(token):
                if hasattr(token, 'tokens'):
                    for sub_token in token.tokens:
                        extract_tables(sub_token)

                if hasattr(token, 'get_name'):
                    name = token.get_name()
                    if name and '.' not in name:  # Simple table name
                        tables.add(name.strip('`"[]'))

            extract_tables(parsed)
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")

        return tables

    def _extract_column_names(self, sql_query: str) -> Dict[str, List[str]]:
        """Extract column names grouped by table."""
        # Simplified implementation - would need more sophisticated parsing
        return {}

    def _count_subquery_depth(self, parsed_query) -> int:
        """Count maximum subquery nesting depth."""

        def count_depth(token, current_depth=0):
            max_depth = current_depth
            if hasattr(token, 'tokens'):
                for sub_token in token.tokens:
                    if str(sub_token).upper().strip().startswith('SELECT'):
                        max_depth = max(max_depth, count_depth(sub_token, current_depth + 1))
                    else:
                        max_depth = max(max_depth, count_depth(sub_token, current_depth))
            return max_depth

        return count_depth(parsed_query)

    def _count_joins(self, sql_query: str) -> int:
        """Count number of JOIN operations in query."""
        join_keywords = ['JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN', 'CROSS JOIN']
        count = 0
        query_upper = sql_query.upper()

        for join_type in join_keywords:
            count += query_upper.count(join_type)

        return count

    async def _audit_validation(
            self,
            user_id: int,
            sql_query: str,
            validation_result: Dict[str, Any]
    ):
        """Audit log the validation result."""
        try:
            await self.audit_logger.log_security_event(
                user_id=user_id,
                event_type="sql_validation",
                details={
                    "query_hash": validation_result["metadata"]["query_hash"],
                    "risk_level": validation_result["risk_level"],
                    "is_safe": validation_result["is_safe"],
                    "threats_detected": validation_result.get("threats_detected", []),
                    "security_score": validation_result.get("security_score", 0)
                },
                risk_level=validation_result["risk_level"]
            )
        except Exception as e:
            logger.error(f"Failed to audit validation: {e}")

    async def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics and statistics."""
        return {
            "total_validations": len(self.validation_cache),
            "cache_hit_rate": 0.0,  # Would calculate based on cache hits
            "active_users": len(self.rate_limiter),
            "threat_patterns": len(sum(self.threat_patterns.values(), [])),
            "security_rules": self.security_rules.copy()
        }