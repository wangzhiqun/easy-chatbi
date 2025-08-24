import re
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import sqlparse
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from sqlparse import sql, tokens as T

from utils import logger


class SQLToolInput(BaseModel):
    query: str = Field(description="SQL query to execute")
    database: Optional[str] = Field(default=None, description="Target database")


class SQLTool(BaseTool):
    name: str = "sql_executor"
    description: str = "Execute SQL queries and return results"
    args_schema = SQLToolInput

    def __init__(self, connector):
        super().__init__()
        self.connector = connector

    def _run(self, query: str, database: Optional[str] = None) -> str:
        try:
            if not ValidationTool.validate_sql_safety(query):
                return "Error: Query contains potentially dangerous operations"

            df = self.connector.execute_query(query)

            if df.empty:
                return "Query executed successfully but returned no results"

            result = f"Query returned {len(df)} rows:\n"
            result += df.head(10).to_string()

            if len(df) > 10:
                result += f"\n... and {len(df) - 10} more rows"

            return result

        except Exception as e:
            logger.error(f"SQL execution failed: {str(e)}")
            return f"Error executing query: {str(e)}"

    async def _arun(self, query: str, database: Optional[str] = None) -> str:
        return self._run(query, database)


class ValidationTool:
    DANGEROUS_KEYWORDS = {
        'DDL': ['DROP', 'CREATE', 'ALTER', 'TRUNCATE'],
        'DML': ['DELETE', 'INSERT', 'UPDATE', 'REPLACE'],
        'DCL': ['GRANT', 'REVOKE'],
        'SYSTEM': ['EXEC', 'EXECUTE', 'CALL'],
        'FILE_OPS': ['LOAD_FILE', 'INTO OUTFILE', 'INTO DUMPFILE'],
    }

    ALLOWED_STATEMENT_TYPES = ['SELECT', 'SHOW', 'DESCRIBE', 'DESC', 'EXPLAIN']

    DANGEROUS_FUNCTIONS = [
        'LOAD_FILE', 'SYSTEM', 'EXEC', 'EXECUTE',
        'SHELL', 'BENCHMARK', 'SLEEP'
    ]

    @classmethod
    def validate_sql_safety(cls, query: str) -> Dict[str, Any]:
        try:
            parsed = sqlparse.parse(query)

            if not parsed:
                return {
                    "valid": False,
                    "error": "无法解析SQL查询",
                    "details": []
                }

            multi_check = cls._check_multiple_statements(parsed)
            if not multi_check["valid"]:
                return multi_check

            for statement in parsed:
                if not cls._is_empty_statement(statement):
                    type_check = cls._check_statement_type(statement)
                    if not type_check["valid"]:
                        return type_check

                    keyword_check = cls._check_dangerous_keywords(statement)
                    if not keyword_check["valid"]:
                        return keyword_check

                    function_check = cls._check_dangerous_functions(statement)
                    if not function_check["valid"]:
                        return function_check

                    injection_check = cls._check_injection_patterns(statement)
                    if not injection_check["valid"]:
                        return injection_check

            return {
                "valid": True,
                "message": "SQL查询通过安全检查",
                "warnings": cls._get_performance_warnings(parsed[0] if parsed else None)
            }

        except Exception as e:
            logger.error(f"SQL validation error: {str(e)}")
            return {
                "valid": False,
                "error": f"验证过程出错: {str(e)}",
                "details": []
            }

    @classmethod
    def _check_multiple_statements(cls, parsed: List[sql.Statement]) -> Dict[str, Any]:
        meaningful_statements = [stmt for stmt in parsed if not cls._is_empty_statement(stmt)]

        if len(meaningful_statements) > 1:
            return {
                "valid": False,
                "error": f"检测到多条SQL语句 ({len(meaningful_statements)} 条)",
                "details": [str(stmt).strip()[:100] + "..." for stmt in meaningful_statements]
            }

        return {"valid": True}

    @classmethod
    def _is_empty_statement(cls, statement: sql.Statement) -> bool:
        content = str(statement).strip()
        if not content:
            return True

        content_no_comments = re.sub(r'--.*$', '', content, flags=re.MULTILINE)
        content_no_comments = re.sub(r'/\*.*?\*/', '', content_no_comments, flags=re.DOTALL)

        return not content_no_comments.strip()

    @classmethod
    def _check_statement_type(cls, statement: sql.Statement) -> Dict[str, Any]:
        try:
            first_token = None
            for token in statement.flatten():
                if token.ttype is T.Keyword and not token.is_whitespace:
                    first_token = token
                    break

            if not first_token:
                return {
                    "valid": False,
                    "error": "无法确定SQL语句类型"
                }

            statement_type = first_token.value.upper()

            if statement_type not in cls.ALLOWED_STATEMENT_TYPES:
                return {
                    "valid": False,
                    "error": f"不允许的SQL语句类型: {statement_type}",
                    "details": f"只允许: {', '.join(cls.ALLOWED_STATEMENT_TYPES)}"
                }

            return {"valid": True}

        except Exception as e:
            return {
                "valid": False,
                "error": f"检查语句类型时出错: {str(e)}"
            }

    @classmethod
    def _check_dangerous_keywords(cls, statement: sql.Statement) -> Dict[str, Any]:
        dangerous_found = []

        for token in statement.flatten():
            if token.ttype is T.Keyword:
                keyword = token.value.upper()

                for category, keywords in cls.DANGEROUS_KEYWORDS.items():
                    if keyword in keywords:
                        dangerous_found.append({
                            "keyword": keyword,
                            "category": category,
                            "position": token
                        })

        if dangerous_found:
            return {
                "valid": False,
                "error": "检测到危险的SQL关键词",
                "details": [f"{item['keyword']} ({item['category']})" for item in dangerous_found]
            }

        return {"valid": True}

    @classmethod
    def _check_dangerous_functions(cls, statement: sql.Statement) -> Dict[str, Any]:
        dangerous_functions_found = []

        for token in statement.flatten():
            if token.ttype is T.Name:
                function_name = token.value.upper()
                if function_name in cls.DANGEROUS_FUNCTIONS:
                    dangerous_functions_found.append(function_name)

        if dangerous_functions_found:
            return {
                "valid": False,
                "error": "检测到危险的SQL函数",
                "details": dangerous_functions_found
            }

        return {"valid": True}

    @classmethod
    def _check_injection_patterns(cls, statement: sql.Statement) -> Dict[str, Any]:
        injection_patterns = []

        stmt_str = str(statement).upper()

        patterns_to_check = [
            (r'\bOR\s+[\'"]?1[\'"]?\s*=\s*[\'"]?1[\'"]?', "经典OR注入"),
            (r'\bAND\s+[\'"]?1[\'"]?\s*=\s*[\'"]?0[\'"]?', "经典AND注入"),
            (r'\bUNION\s+(?:ALL\s+)?SELECT\b', "UNION注入"),
            (r';\s*--', "注释注入"),
            (r'\bINTO\s+(?:OUT|DUMP)FILE\b', "文件写入"),
            (r'\bLOAD_FILE\s*\(', "文件读取"),
        ]

        for pattern, description in patterns_to_check:
            if re.search(pattern, stmt_str):
                if not cls._is_in_string_literal(statement, pattern):
                    injection_patterns.append(description)

        if injection_patterns:
            return {
                "valid": False,
                "error": "检测到疑似SQL注入模式",
                "details": injection_patterns
            }

        return {"valid": True}

    @classmethod
    def _is_in_string_literal(cls, statement: sql.Statement, pattern: str) -> bool:
        try:
            string_literals = []
            for token in statement.flatten():
                if token.ttype in (T.String.Single, T.String.Symbol):
                    string_literals.append(token.value.upper())

            for literal in string_literals:
                if re.search(pattern, literal):
                    return True

            return False

        except Exception:
            return False

    @classmethod
    def _get_performance_warnings(cls, statement: Optional[sql.Statement]) -> List[str]:
        warnings = []

        if not statement:
            return warnings

        stmt_str = str(statement).upper()

        if 'SELECT *' in stmt_str:
            warnings.append("建议指定具体字段名而不是使用 *")

        if 'SELECT' in stmt_str and 'LIMIT' not in stmt_str:
            warnings.append("建议添加 LIMIT 子句以提高查询性能")

        if 'SELECT' in stmt_str and 'WHERE' not in stmt_str and 'LIMIT' not in stmt_str:
            warnings.append("查询没有WHERE条件，可能返回大量数据")

        return warnings

    @classmethod
    def validate_sql_syntax(cls, query: str) -> Tuple[bool, Optional[str]]:
        try:
            parsed = sqlparse.parse(query)

            if not parsed:
                return False, "无法解析SQL查询"

            valid_statements = [stmt for stmt in parsed if not cls._is_empty_statement(stmt)]

            if not valid_statements:
                return False, "没有发现有效的SQL语句"

            for statement in valid_statements:
                syntax_check = cls._validate_statement_syntax(statement)
                if not syntax_check[0]:
                    return syntax_check

            return True, None

        except Exception as e:
            return False, f"语法验证失败: {str(e)}"

    @classmethod
    def _validate_statement_syntax(cls, statement: sql.Statement) -> Tuple[bool, Optional[str]]:
        try:
            paren_count = 0
            quote_count = 0
            in_string = False
            quote_char = None

            for token in statement.flatten():
                if token.ttype in (T.String.Single, T.String.Symbol):
                    continue

                token_value = token.value

                for char in token_value:
                    if not in_string:
                        if char in ("'", '"'):
                            in_string = True
                            quote_char = char
                            quote_count += 1
                        elif char == '(':
                            paren_count += 1
                        elif char == ')':
                            paren_count -= 1
                    else:
                        if char == quote_char:
                            in_string = False
                            quote_char = None

            if paren_count != 0:
                return False, "括号不匹配"

            if in_string:
                return False, "引号不匹配"

            return True, None

        except Exception as e:
            return False, f"语法检查失败: {str(e)}"

    @classmethod
    def get_statement_info(cls, query: str) -> Dict[str, Any]:
        try:
            parsed = sqlparse.parse(query)

            if not parsed:
                return {"error": "无法解析SQL"}

            info = {
                "statement_count": len([s for s in parsed if not cls._is_empty_statement(s)]),
                "statements": []
            }

            for statement in parsed:
                if not cls._is_empty_statement(statement):
                    stmt_info = {
                        "type": cls._get_statement_type(statement),
                        "tokens": len(list(statement.flatten())),
                        "tables": cls._extract_tables(statement),
                        "columns": cls._extract_columns(statement),
                        "has_where": 'WHERE' in str(statement).upper(),
                        "has_limit": 'LIMIT' in str(statement).upper(),
                        "has_order_by": 'ORDER BY' in str(statement).upper()
                    }
                    info["statements"].append(stmt_info)

            return info

        except Exception as e:
            return {"error": f"分析失败: {str(e)}"}

    @classmethod
    def _get_statement_type(cls, statement: sql.Statement) -> str:
        for token in statement.flatten():
            if token.ttype is T.Keyword and not token.is_whitespace:
                return token.value.upper()
        return "UNKNOWN"

    @classmethod
    def _extract_tables(cls, statement: sql.Statement) -> List[str]:
        tables = []
        tokens = list(statement.flatten())

        for i, token in enumerate(tokens):
            if (token.ttype is T.Keyword and
                    token.value.upper() in ('FROM', 'JOIN', 'UPDATE', 'INTO')):
                for j in range(i + 1, len(tokens)):
                    next_token = tokens[j]
                    if next_token.ttype is T.Name and not next_token.is_whitespace:
                        tables.append(next_token.value)
                        break
                    elif next_token.ttype is T.Keyword:
                        break

        return list(set(tables))

    @classmethod
    def _extract_columns(cls, statement: sql.Statement) -> List[str]:
        columns = []
        in_select = False

        for token in statement.flatten():
            if token.ttype is T.Keyword and token.value.upper() == 'SELECT':
                in_select = True
                continue
            elif token.ttype is T.Keyword and token.value.upper() == 'FROM':
                break
            elif in_select and token.ttype is T.Name:
                columns.append(token.value)

        return columns

    @classmethod
    def sanitize_input(cls, user_input: str) -> str:
        sanitized = user_input.replace("'", "''")
        sanitized = re.sub(r'[;\-\-\/\*\*\/]', '', sanitized)
        sanitized = sanitized.strip()

        return sanitized


class AnalysisTool:

    @staticmethod
    def generate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        stats = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }

        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            stats['numeric_summary'] = df[numeric_cols].describe().to_dict()

        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            cat_stats = {}
            for col in cat_cols:
                cat_stats[col] = {
                    'unique_count': df[col].nunique(),
                    'top_value': df[col].mode()[0] if not df[col].mode().empty else None,
                    'top_frequency': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
                }
            stats['categorical_summary'] = cat_stats

        return stats

    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> List[str]:
        patterns = []

        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                patterns.append(f"Time series data detected in column: {col}")

        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr.append(
                            f"{corr_matrix.columns[i]} and {corr_matrix.columns[j]}: "
                            f"{corr_matrix.iloc[i, j]:.2f}"
                        )
            if high_corr:
                patterns.append(f"High correlation detected: {', '.join(high_corr)}")

        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((numeric_df[col] < (Q1 - 1.5 * IQR)) |
                        (numeric_df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                patterns.append(f"Outliers detected in {col}: {outliers} values")

        return patterns

    @staticmethod
    def suggest_visualizations(df: pd.DataFrame) -> List[Dict[str, Any]]:
        suggestions = []

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]

        if date_cols and numeric_cols:
            suggestions.append({
                'type': 'line',
                'title': 'Time Series Analysis',
                'x': date_cols[0],
                'y': numeric_cols[0],
                'description': 'Track trends over time'
            })

        if numeric_cols:
            suggestions.append({
                'type': 'histogram',
                'title': 'Distribution Analysis',
                'column': numeric_cols[0],
                'description': 'Understand data distribution'
            })

        if cat_cols and numeric_cols:
            suggestions.append({
                'type': 'bar',
                'title': 'Category Comparison',
                'x': cat_cols[0],
                'y': numeric_cols[0],
                'description': 'Compare values across categories'
            })

        if len(numeric_cols) > 2:
            suggestions.append({
                'type': 'heatmap',
                'title': 'Correlation Matrix',
                'columns': numeric_cols,
                'description': 'Identify relationships between variables'
            })

        return suggestions
