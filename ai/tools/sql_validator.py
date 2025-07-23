"""
SQL Validator for ChatBI platform.
Validates SQL queries for syntax, security, and best practices.
"""

import re
import sqlparse
from typing import Dict, List, Any, Set, Tuple, Optional
from sqlparse.sql import IdentifierList, Identifier, Function
from sqlparse.tokens import Keyword, Name

from utils.logger import get_logger
from utils.exceptions import SQLSecurityException, ValidationException, ErrorCodes

logger = get_logger(__name__)


class SQLValidator:
    """
    Comprehensive SQL validator that checks for security issues,
    syntax correctness, and adherence to best practices.
    """

    def __init__(self):
        """Initialize SQL validator with security rules and patterns."""

        # Dangerous SQL keywords that should never be allowed
        self.forbidden_keywords = {
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'CREATE', 'ALTER',
            'TRUNCATE', 'REPLACE', 'MERGE', 'GRANT', 'REVOKE',
            'EXEC', 'EXECUTE', 'CALL', 'LOAD', 'OUTFILE', 'INFILE'
        }

        # System tables that should not be accessible
        self.forbidden_tables = {
            'information_schema', 'mysql', 'performance_schema',
            'sys', 'pg_catalog', 'pg_stat', 'sqlite_master'
        }

        # Suspicious patterns that might indicate SQL injection
        self.injection_patterns = [
            r"';.*--",  # Comment after semicolon
            r"union.*select",  # UNION-based injection
            r"or.*1=1",  # Boolean-based injection
            r"and.*1=2",  # Boolean-based injection
            r"exec\s*\(",  # Function execution
            r"char\s*\(",  # Character concatenation
            r"concat\s*\(",  # String concatenation attacks
            r"load_file\s*\(",  # File reading functions
            r"into\s+outfile",  # File writing
            r"benchmark\s*\(",  # Time-based attacks
            r"sleep\s*\(",  # Time delays
            r"waitfor\s+delay",  # SQL Server delays
        ]

        # Function patterns that should be restricted
        self.restricted_functions = {
            'LOAD_FILE', 'INTO OUTFILE', 'INTO DUMPFILE', 'BENCHMARK',
            'SLEEP', 'GET_LOCK', 'RELEASE_LOCK', 'FOUND_ROWS',
            'ROW_COUNT', 'USER', 'DATABASE', 'VERSION'
        }

        # Maximum query complexity limits
        self.max_subqueries = 5
        self.max_joins = 10
        self.max_where_conditions = 20
        self.max_query_length = 5000

    async def validate_query(
            self,
            sql_query: str,
            available_tables: List[Dict[str, Any]],
            user_permissions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive validation of SQL query.

        Args:
            sql_query: SQL query to validate
            available_tables: List of tables user has access to
            user_permissions: User's specific permissions

        Returns:
            Validation result with safety status and recommendations
        """
        logger.info("Starting SQL validation")

        validation_result = {
            "is_safe": True,
            "status": "safe",
            "issues": [],
            "warnings": [],
            "suggestions": [],
            "query_info": {},
            "security_score": 100
        }

        try:
            # Step 1: Basic security checks
            security_issues = self._check_security_violations(sql_query)
            if security_issues:
                validation_result["is_safe"] = False
                validation_result["status"] = "unsafe"
                validation_result["issues"].extend(security_issues)
                validation_result["security_score"] -= len(security_issues) * 30

            # Step 2: Parse and analyze query structure
            try:
                parsed_query = sqlparse.parse(sql_query)[0]
                query_analysis = self._analyze_query_structure(parsed_query)
                validation_result["query_info"] = query_analysis
            except Exception as e:
                validation_result["issues"].append(f"SQL parsing failed: {str(e)}")
                validation_result["is_safe"] = False
                validation_result["status"] = "syntax_error"
                return validation_result

            # Step 3: Check table access permissions
            table_issues = self._validate_table_access(query_analysis, available_tables)
            if table_issues:
                validation_result["issues"].extend(table_issues)
                validation_result["is_safe"] = False
                validation_result["status"] = "access_denied"
                validation_result["security_score"] -= len(table_issues) * 20

            # Step 4: Check query complexity
            complexity_issues = self._check_query_complexity(query_analysis)
            if complexity_issues:
                validation_result["warnings"].extend(complexity_issues)
                validation_result["security_score"] -= len(complexity_issues) * 10

            # Step 5: Performance and best practice checks
            performance_suggestions = self._check_performance_issues(sql_query, query_analysis)
            validation_result["suggestions"].extend(performance_suggestions)

            # Step 6: Final safety assessment
            if validation_result["security_score"] < 50:
                validation_result["is_safe"] = False
                validation_result["status"] = "high_risk"
            elif validation_result["security_score"] < 80:
                validation_result["status"] = "medium_risk"
                validation_result["warnings"].append("Query has some security concerns")

            logger.info(f"SQL validation completed: {validation_result['status']}")
            return validation_result

        except Exception as e:
            logger.error(f"SQL validation failed: {e}")
            return {
                "is_safe": False,
                "status": "validation_error",
                "issues": [f"Validation process failed: {str(e)}"],
                "warnings": [],
                "suggestions": [],
                "query_info": {},
                "security_score": 0
            }

    def _check_security_violations(self, sql_query: str) -> List[str]:
        """Check for obvious security violations."""
        issues = []
        query_upper = sql_query.upper()
        query_lower = sql_query.lower()

        # Check for forbidden keywords
        for keyword in self.forbidden_keywords:
            if re.search(rf'\b{keyword}\b', query_upper):
                issues.append(f"Forbidden operation detected: {keyword}")

        # Check for SQL injection patterns
        for pattern in self.injection_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                issues.append(f"Potential SQL injection pattern detected: {pattern}")

        # Check for restricted functions
        for func in self.restricted_functions:
            if re.search(rf'\b{func}\s*\(', query_upper):
                issues.append(f"Restricted function detected: {func}")

        # Check for system table access
        for table in self.forbidden_tables:
            if re.search(rf'\b{table}\b', query_lower):
                issues.append(f"System table access detected: {table}")

        # Check for suspicious character sequences
        if '--' in sql_query and not sql_query.strip().startswith('--'):
            issues.append("SQL comment detected in middle of query")

        if ';' in sql_query.rstrip(';'):  # Semicolon not at the end
            issues.append("Multiple statements detected (semicolon in query)")

        # Check for quote escaping attempts
        if "\\'" in sql_query or '\\"' in sql_query:
            issues.append("Quote escaping detected")

        return issues

    def _analyze_query_structure(self, parsed_query) -> Dict[str, Any]:
        """Analyze the structure of the parsed SQL query."""
        analysis = {
            "statement_type": None,
            "tables": [],
            "columns": [],
            "functions": [],
            "subqueries": 0,
            "joins": 0,
            "where_conditions": 0,
            "order_by": False,
            "group_by": False,
            "having": False,
            "limit": None
        }

        # Get statement type
        analysis["statement_type"] = parsed_query.get_type()

        # Extract tables, columns, and other elements
        self._extract_query_elements(parsed_query, analysis)

        return analysis

    def _extract_query_elements(self, token, analysis: Dict[str, Any]):
        """Recursively extract elements from parsed query."""
        if hasattr(token, 'tokens'):
            for sub_token in token.tokens:
                self._extract_query_elements(sub_token, analysis)

        # Check token type and extract information
        if token.ttype is Keyword:
            keyword = token.value.upper()
            if keyword == 'ORDER':
                analysis["order_by"] = True
            elif keyword == 'GROUP':
                analysis["group_by"] = True
            elif keyword == 'HAVING':
                analysis["having"] = True
            elif keyword in ['JOIN', 'INNER', 'LEFT', 'RIGHT', 'FULL']:
                analysis["joins"] += 1

        elif isinstance(token, IdentifierList):
            for identifier in token.get_identifiers():
                if isinstance(identifier, Identifier):
                    analysis["columns"].append(str(identifier))

        elif isinstance(token, Identifier):
            # Could be table or column
            name = str(token)
            if '.' not in name:  # Likely a table name
                analysis["tables"].append(name)

        elif isinstance(token, Function):
            analysis["functions"].append(str(token))

        # Count subqueries
        if hasattr(token, 'tokens'):
            query_text = str(token).upper()
            analysis["subqueries"] += query_text.count('SELECT') - 1  # Subtract main query

        # Look for LIMIT
        if token.ttype is Keyword and token.value.upper() == 'LIMIT':
            # Next meaningful token should be the limit value
            pass  # Implementation would extract the limit value

    def _validate_table_access(
            self,
            query_analysis: Dict[str, Any],
            available_tables: List[Dict[str, Any]]
    ) -> List[str]:
        """Validate that user has access to all referenced tables."""
        issues = []

        # Get list of available table names
        available_table_names = {table.get("name", "").lower() for table in available_tables}

        # Check each table in the query
        for table in query_analysis.get("tables", []):
            table_name = table.lower().strip('`"[]')  # Remove common quote characters

            # Skip if it's an alias or complex expression
            if ' ' in table_name or '(' in table_name:
                continue

            if table_name not in available_table_names:
                issues.append(f"Access denied to table: {table}")

        return issues

    def _check_query_complexity(self, query_analysis: Dict[str, Any]) -> List[str]:
        """Check if query complexity exceeds safe limits."""
        warnings = []

        # Check subquery count
        subqueries = query_analysis.get("subqueries", 0)
        if subqueries > self.max_subqueries:
            warnings.append(f"Too many subqueries ({subqueries} > {self.max_subqueries})")

        # Check join count
        joins = query_analysis.get("joins", 0)
        if joins > self.max_joins:
            warnings.append(f"Too many joins ({joins} > {self.max_joins})")

        # Check WHERE condition complexity (approximate)
        where_conditions = query_analysis.get("where_conditions", 0)
        if where_conditions > self.max_where_conditions:
            warnings.append(f"Complex WHERE clause ({where_conditions} conditions)")

        return warnings

    def _check_performance_issues(
            self,
            sql_query: str,
            query_analysis: Dict[str, Any]
    ) -> List[str]:
        """Check for potential performance issues and suggest improvements."""
        suggestions = []

        # Check for LIMIT clause
        if not query_analysis.get("limit") and 'LIMIT' not in sql_query.upper():
            suggestions.append("Consider adding LIMIT clause to prevent large result sets")

        # Check for SELECT *
        if 'SELECT *' in sql_query.upper():
            suggestions.append("Consider selecting specific columns instead of SELECT *")

        # Check for ORDER BY without LIMIT
        if query_analysis.get("order_by") and not query_analysis.get("limit"):
            suggestions.append("ORDER BY without LIMIT may be inefficient for large datasets")

        # Check for functions in WHERE clause
        where_functions = ['UPPER(', 'LOWER(', 'SUBSTRING(', 'DATE(']
        query_upper = sql_query.upper()
        for func in where_functions:
            if func in query_upper and 'WHERE' in query_upper:
                suggestions.append(f"Function {func} in WHERE clause may prevent index usage")

        # Check for multiple JOINs without apparent relationships
        if query_analysis.get("joins", 0) > 3:
            suggestions.append("Multiple joins detected - ensure proper indexing on join columns")

        # Check for LIKE patterns that start with wildcard
        if re.search(r"LIKE\s+['\"]%", sql_query, re.IGNORECASE):
            suggestions.append("LIKE patterns starting with % cannot use indexes efficiently")

        return suggestions

    def get_validation_rules(self) -> Dict[str, Any]:
        """Get current validation rules and limits."""
        return {
            "forbidden_keywords": list(self.forbidden_keywords),
            "forbidden_tables": list(self.forbidden_tables),
            "complexity_limits": {
                "max_subqueries": self.max_subqueries,
                "max_joins": self.max_joins,
                "max_where_conditions": self.max_where_conditions,
                "max_query_length": self.max_query_length
            },
            "restricted_functions": list(self.restricted_functions)
        }

    def is_select_only(self, sql_query: str) -> bool:
        """Check if query is a safe SELECT-only statement."""
        try:
            parsed = sqlparse.parse(sql_query)[0]
            statement_type = parsed.get_type()
            return statement_type == 'SELECT'
        except:
            return False

    def extract_table_names(self, sql_query: str) -> Set[str]:
        """Extract all table names referenced in the query."""
        try:
            parsed = sqlparse.parse(sql_query)[0]
            tables = set()

            def extract_tables(token):
                if hasattr(token, 'tokens'):
                    for sub_token in token.tokens:
                        extract_tables(sub_token)

                # Look for table names after FROM and JOIN keywords
                if isinstance(token, Identifier):
                    tables.add(str(token).strip('`"[]'))

            extract_tables(parsed)
            return tables
        except:
            return set()

    def sanitize_query(self, sql_query: str) -> str:
        """Basic query sanitization (remove comments, normalize whitespace)."""
        # Remove SQL comments
        query = re.sub(r'--.*$', '', sql_query, flags=re.MULTILINE)
        query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)

        # Normalize whitespace
        query = ' '.join(query.split())

        # Remove trailing semicolon
        query = query.rstrip(';')

        return query.strip()