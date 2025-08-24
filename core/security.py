import hashlib
import re
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Tuple

import jwt
import sqlparse
from sqlparse.sql import Statement, Token
from sqlparse.tokens import Keyword, Comment, Name

from utils import logger, get_config, ValidationError


class SecurityManager:

    def __init__(self):
        self.config = get_config()
        self.dangerous_keywords = self._load_dangerous_keywords()
        self.allowed_functions = self._load_allowed_functions()
        self.dangerous_functions = self._load_dangerous_functions()
        self.rate_limits = {}
        logger.info("Initialized Security Manager")

    def _load_dangerous_keywords(self) -> Set[str]:
        return {
            'DROP', 'CREATE', 'ALTER', 'TRUNCATE', 'REPLACE',

            'INSERT', 'UPDATE', 'DELETE', 'MERGE',

            'GRANT', 'REVOKE',

            'COMMIT', 'ROLLBACK', 'SAVEPOINT',

            'CALL', 'EXEC', 'EXECUTE',

            'LOAD', 'INTO OUTFILE', 'INTO DUMPFILE'
        }

    def _load_allowed_functions(self) -> Set[str]:
        return {
            'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'GROUP_CONCAT',

            'CONCAT', 'SUBSTRING', 'LENGTH', 'UPPER', 'LOWER', 'TRIM',
            'LEFT', 'RIGHT', 'REPLACE', 'REGEXP_REPLACE',

            'NOW', 'CURDATE', 'CURTIME', 'DATE', 'TIME', 'YEAR', 'MONTH', 'DAY',
            'DATE_FORMAT', 'STR_TO_DATE', 'DATEDIFF', 'DATE_ADD', 'DATE_SUB',

            'ROUND', 'CEIL', 'FLOOR', 'ABS', 'MOD', 'POWER', 'SQRT',

            'IF', 'CASE', 'COALESCE', 'NULLIF', 'IFNULL',

            'CAST', 'CONVERT',
        }

    def _load_dangerous_functions(self) -> Set[str]:
        return {
            'LOAD_FILE', 'INTO_OUTFILE', 'INTO_DUMPFILE',

            'BENCHMARK', 'SLEEP', 'GET_LOCK', 'RELEASE_LOCK',

            'USER', 'CURRENT_USER', 'SESSION_USER', 'SYSTEM_USER',
            'DATABASE', 'SCHEMA', 'VERSION',

            'xp_cmdshell', 'sp_configure', 'openrowset', 'opendatasource',

            'LOAD_DATA', 'SELECT_INTO_OUTFILE',
        }

    def validate_sql_query(self, query: str) -> tuple[bool, Optional[str]]:

        try:
            if not query or not query.strip():
                return False, "Query cannot be empty"

            if len(query) > 10000:
                return False, "Query too long (max 10000 characters)"

            try:
                parsed_statements = sqlparse.parse(query)
            except Exception as e:
                return False, f"SQL parsing failed: {str(e)}"

            if not parsed_statements:
                return False, "No valid SQL statements found"

            for i, statement in enumerate(parsed_statements):
                is_valid, error = self._validate_statement(statement, i)
                if not is_valid:
                    return False, error

            is_valid, error = self._additional_security_checks(query)
            if not is_valid:
                return False, error

            logger.debug("SQL query validated successfully")
            return True, None

        except Exception as e:
            logger.error(f"SQL validation error: {str(e)}")
            return False, f"Validation error: {str(e)}"

    def _validate_statement(self, statement: Statement, index: int) -> Tuple[bool, Optional[str]]:
        try:
            tokens = [token for token in statement.flatten()
                      if not token.is_whitespace and token.ttype not in (Comment.Single, Comment.Multiline)]

            if not tokens:
                return False, f"Statement {index + 1}: No meaningful tokens found"

            first_token = tokens[0]
            if (first_token.ttype is Keyword and
                    first_token.value.upper() not in ['SELECT', 'WITH']):
                return False, f"Statement {index + 1}: Only SELECT and WITH statements are allowed, found: {first_token.value}"

            for token in tokens:
                if token.ttype is Keyword:
                    keyword = token.value.upper()
                    if keyword in self.dangerous_keywords:
                        if keyword in ['SHOW', 'DESCRIBE', 'DESC', 'EXPLAIN']:
                            continue
                        if keyword == 'REPLACE' and self._is_replace_function(token, tokens):
                            continue
                        return False, f"Statement {index + 1}: Dangerous keyword detected: {keyword}"

            function_check = self._check_functions(tokens)
            if not function_check[0]:
                return False, f"Statement {index + 1}: {function_check[1]}"

            union_check = self._check_union_injection(tokens)
            if not union_check[0]:
                return False, f"Statement {index + 1}: {union_check[1]}"

            return True, None

        except Exception as e:
            return False, f"Statement {index + 1}: Validation error: {str(e)}"

    def _is_replace_function(self, token: Token, all_tokens: List[Token]) -> bool:
        try:
            token_index = all_tokens.index(token)
            for i in range(token_index + 1, min(token_index + 3, len(all_tokens))):
                if all_tokens[i].value == '(':
                    return True
            return False
        except:
            return False

    def _check_functions(self, tokens: List[Token]) -> Tuple[bool, Optional[str]]:
        for i, token in enumerate(tokens):
            if token.ttype in (Name, Name.Builtin) or (token.ttype is None and token.value.isalpha()):
                func_name = token.value.upper()

                if (i + 1 < len(tokens) and tokens[i + 1].value == '(' and
                        func_name in self.dangerous_functions):
                    return False, f"Dangerous function detected: {func_name}"

        return True, None

    def _check_union_injection(self, tokens: List[Token]) -> Tuple[bool, Optional[str]]:
        for i, token in enumerate(tokens):
            if token.ttype is Keyword and token.value.upper() == 'UNION':
                for j in range(i + 1, min(i + 3, len(tokens))):
                    if tokens[j].ttype is Keyword and tokens[j].value.upper() == 'SELECT':
                        return False, "UNION SELECT detected - potentially dangerous"

        return True, None

    def _additional_security_checks(self, query: str) -> Tuple[bool, Optional[str]]:

        dangerous_patterns = [
            (r"(;.*;\s*$)", "Multiple statements detected"),
            (r"(\bINTO\s+OUTFILE\b)", "File output operations not allowed"),
            (r"(\bLOAD_FILE\s*\()", "File loading operations not allowed"),
            (r"(\b0x[0-9a-fA-F]+\b)", "Hex literals potentially dangerous"),
            (r"(xp_cmdshell|sp_configure)", "System procedures not allowed"),
        ]

        for pattern, message in dangerous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, message

        clean_query = self._remove_comments(query)

        single_quotes = clean_query.count("'")
        double_quotes = clean_query.count('"')

        if single_quotes % 2 != 0:
            return False, "Unbalanced single quotes"

        if double_quotes % 2 != 0:
            return False, "Unbalanced double quotes"

        open_parens = clean_query.count("(")
        close_parens = clean_query.count(")")

        if open_parens != close_parens:
            return False, "Unbalanced parentheses"

        return True, None

    def _remove_comments(self, query: str) -> str:
        try:
            parsed = sqlparse.parse(query)
            result = []

            for statement in parsed:
                for token in statement.flatten():
                    if token.ttype not in (Comment.Single, Comment.Multiline):
                        result.append(str(token))

            return ''.join(result)
        except:
            return query

    def is_read_only_query(self, query: str) -> bool:
        try:
            parsed_statements = sqlparse.parse(query)

            for statement in parsed_statements:
                tokens = [token for token in statement.flatten()
                          if not token.is_whitespace and token.ttype not in (Comment.Single, Comment.Multiline)]

                if not tokens:
                    continue

                first_token = tokens[0]
                if (first_token.ttype is Keyword and
                        first_token.value.upper() not in ['SELECT', 'WITH']):
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking read-only query: {str(e)}")
            return False

    def get_query_info(self, query: str) -> dict:
        try:
            parsed_statements = sqlparse.parse(query)
            info = {
                'statement_count': len(parsed_statements),
                'has_comments': False,
                'query_type': 'UNKNOWN',
                'tables_mentioned': [],
                'functions_used': [],
                'is_read_only': True
            }

            for statement in parsed_statements:
                for token in statement.flatten():
                    if token.ttype in (Comment.Single, Comment.Multiline):
                        info['has_comments'] = True
                        break

                tokens = [token for token in statement.flatten()
                          if not token.is_whitespace and token.ttype not in (Comment.Single, Comment.Multiline)]

                if tokens:
                    first_token = tokens[0]
                    if first_token.ttype is Keyword:
                        info['query_type'] = first_token.value.upper()
                        if first_token.value.upper() not in ['SELECT', 'WITH']:
                            info['is_read_only'] = False

            return info

        except Exception as e:
            logger.error(f"Error getting query info: {str(e)}")
            return {'error': str(e)}

    def sanitize_input(self, input_str: str) -> str:
        sanitized = input_str.replace('\x00', '')

        sanitized = sanitized.replace("'", "''")

        sanitized = re.sub(r'--.*$', '', sanitized, flags=re.MULTILINE)
        sanitized = re.sub(r'/\*.*?\*/', '', sanitized, flags=re.DOTALL)

        max_length = 1000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        return sanitized.strip()

    def generate_token(self, user_id: str, additional_claims: Optional[Dict] = None) -> str:
        try:
            payload = {
                'user_id': user_id,
                'exp': datetime.utcnow() + timedelta(minutes=self.config.access_token_expire_minutes),
                'iat': datetime.utcnow(),
                'jti': secrets.token_urlsafe(16)
            }

            if additional_claims:
                payload.update(additional_claims)

            token = jwt.encode(
                payload,
                self.config.secret_key,
                algorithm=self.config.algorithm
            )

            logger.info(f"Generated token for user: {user_id}")
            return token

        except Exception as e:
            logger.error(f"Token generation failed: {str(e)}")
            raise ValidationError(f"Token generation failed: {str(e)}")

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm]
            )

            logger.debug(f"Token verified for user: {payload.get('user_id')}")
            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {str(e)}")
            return None

    def hash_password(self, password: str) -> str:
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return f"{salt}${password_hash.hex()}"

    def verify_password(self, password: str, hashed: str) -> bool:
        try:
            salt, password_hash = hashed.split('$')
            test_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            return test_hash.hex() == password_hash
        except Exception as e:
            logger.error(f"Password verification failed: {str(e)}")
            return False

    def check_rate_limit(
            self,
            identifier: str,
            max_requests: int = 100,
            window_seconds: int = 60
    ) -> bool:
        now = datetime.now()
        window_start = now - timedelta(seconds=window_seconds)

        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = []

        self.rate_limits[identifier] = [
            timestamp for timestamp in self.rate_limits[identifier]
            if timestamp > window_start
        ]

        if len(self.rate_limits[identifier]) >= max_requests:
            logger.warning(f"Rate limit exceeded for: {identifier}")
            return False

        self.rate_limits[identifier].append(now)
        return True

    def validate_table_access(self, table_name: str, allowed_tables: List[str]) -> bool:

        table_name = re.sub(r'[^\w\.]', '', table_name)

        if allowed_tables and table_name not in allowed_tables:
            logger.warning(f"Access denied to table: {table_name}")
            return False

        system_prefixes = ['sys', 'information_schema', 'mysql', 'performance_schema']
        for prefix in system_prefixes:
            if table_name.lower().startswith(prefix):
                logger.warning(f"Access denied to system table: {table_name}")
                return False

        return True

    def mask_sensitive_data(self, data: Dict[str, Any], sensitive_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        if not sensitive_fields:
            sensitive_fields = [
                'password', 'pwd', 'secret', 'token', 'api_key',
                'credit_card', 'ssn', 'email', 'phone'
            ]

        masked_data = data.copy()

        for key, value in masked_data.items():
            if any(field in key.lower() for field in sensitive_fields):
                if isinstance(value, str):
                    if len(value) > 4:
                        masked_data[key] = value[:2] + '*' * (len(value) - 4) + value[-2:]
                    else:
                        masked_data[key] = '*' * len(value)
                elif isinstance(value, (int, float)):
                    masked_data[key] = '***'
                elif isinstance(value, dict):
                    masked_data[key] = self.mask_sensitive_data(value, sensitive_fields)

        return masked_data

    def generate_api_key(self) -> str:
        return f"sk_{secrets.token_urlsafe(32)}"

    def validate_api_key(self, api_key: str) -> bool:
        pattern = r'^sk_[A-Za-z0-9_-]{43}$'
        return bool(re.match(pattern, api_key))
