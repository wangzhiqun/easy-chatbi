import re
from typing import Dict, Any, Optional, Tuple

from utils import logger, AIError
from ..llm_client import LLMClient
from ..prompts import PromptTemplates
from ..tools import ValidationTool


class SQLAgent:

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self.validator = ValidationTool()
        logger.info("Initialized SQL Agent")

    def generate_sql(
            self,
            question: str,
            schema_info: Dict[str, Any],
            examples: Optional[list] = None
    ) -> Tuple[str, str]:
        try:
            system_prompt, user_prompt = PromptTemplates.get_sql_prompt(
                question=question,
                schema_info=schema_info
            )

            if examples:
                user_prompt += "\n\n参考示例：\n"
                for ex in examples:
                    user_prompt += f"问题：{ex['question']}\nSQL：{ex['sql']}\n\n"

            response = self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3
            )

            sql_query = self._extract_sql(response)

            if not self.validator.validate_sql_safety(sql_query):
                raise AIError("Generated SQL contains dangerous operations")

            is_valid, error = self.validator.validate_sql_syntax(sql_query)
            if not is_valid:
                logger.warning(f"SQL syntax validation failed: {error}")
                sql_query = self._fix_sql(sql_query, error, schema_info)

            logger.info(f"Generated SQL query: {sql_query[:100]}...")
            return sql_query, response

        except Exception as e:
            logger.error(f"SQL generation failed: {str(e)}")
            raise AIError(f"SQL generation failed: {str(e)}")

    def optimize_sql(self, sql_query: str, schema_info: Dict[str, Any]) -> str:
        try:
            prompt = f"""
优化以下SQL查询以提高性能：

原始SQL：
{sql_query}

数据库架构：
{self._format_schema_brief(schema_info)}

请提供优化后的SQL，考虑：
1. 使用合适的索引
2. 避免全表扫描
3. 优化JOIN顺序
4. 使用合适的聚合函数

输出优化后的SQL：
"""

            system_prompt = "你是一个SQL性能优化专家，精通查询优化技术。"

            response = self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3
            )

            optimized_sql = self._extract_sql(response)
            logger.info("SQL query optimized successfully")
            return optimized_sql

        except Exception as e:
            logger.error(f"SQL optimization failed: {str(e)}")
            return sql_query

    def explain_sql(self, sql_query: str) -> str:
        try:
            prompt = f"""
请用简单的中文解释以下SQL查询的作用：

SQL查询：
{sql_query}

解释应该包括：
1. 查询的主要目的
2. 涉及的数据表
3. 筛选条件
4. 返回的结果

用通俗易懂的语言解释：
"""

            response = self.llm.generate(
                prompt=prompt,
                temperature=0.5
            )

            return response

        except Exception as e:
            logger.error(f"SQL explanation failed: {str(e)}")
            return "无法解释此SQL查询"

    def _extract_sql(self, response: str) -> str:

        sql_pattern = r'```sql\n(.*?)\n```'
        matches = re.findall(sql_pattern, response, re.DOTALL | re.IGNORECASE)

        if matches:
            return matches[0].strip()

        select_pattern = r'(SELECT\s+.*?(?:;|$))'
        matches = re.findall(select_pattern, response, re.DOTALL | re.IGNORECASE)

        if matches:
            return matches[0].strip()

        lines = response.split('\n')
        sql_lines = []
        in_sql = False

        for line in lines:
            if 'SELECT' in line.upper():
                in_sql = True
            if in_sql:
                sql_lines.append(line)
                if ';' in line:
                    break

        if sql_lines:
            return '\n'.join(sql_lines).strip()

        return response.strip()

    def _fix_sql(
            self,
            sql_query: str,
            error: str,
            schema_info: Dict[str, Any]
    ) -> str:
        try:
            prompt = f"""
修复以下SQL查询的语法错误：

错误的SQL：
{sql_query}

错误信息：
{error}

数据库架构：
{self._format_schema_brief(schema_info)}

请提供修正后的SQL查询：
"""

            response = self.llm.generate(
                prompt=prompt,
                temperature=0.2
            )

            fixed_sql = self._extract_sql(response)
            logger.info("SQL query fixed successfully")
            return fixed_sql

        except Exception as e:
            logger.error(f"SQL fix failed: {str(e)}")
            return sql_query

    def _format_schema_brief(self, schema_info: Dict[str, Any]) -> str:
        tables = schema_info.get('tables', {})
        lines = []

        for table_name, table_info in list(tables.items())[:5]:
            columns = table_info.get('columns', [])
            col_names = [col['COLUMN_NAME'] for col in columns[:8]]
            lines.append(f"{table_name}: {', '.join(col_names)}")

        return '\n'.join(lines)
