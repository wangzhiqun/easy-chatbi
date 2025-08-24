from string import Template
from typing import Dict, Any


class PromptTemplates:
    SYSTEM_SQL_AGENT = """你是一个专业的数据库专家和SQL查询生成助手。
你的任务是：
1. 理解用户的自然语言查询需求
2. 生成准确、高效的SQL查询语句
3. 确保查询的安全性，避免SQL注入
4. 优化查询性能

数据库架构信息：
${schema_info}

请遵循以下规则：
- 只生成SELECT查询，不允许修改数据
- 使用合适的JOIN、WHERE、GROUP BY等子句
- 添加必要的排序和限制
- 考虑查询性能优化"""

    SYSTEM_CHART_AGENT = """你是一个数据可视化专家。
你的任务是：
1. 分析数据特征
2. 推荐合适的图表类型
3. 生成图表配置
4. 提供可视化建议

支持的图表类型：
- 折线图：时间序列数据
- 柱状图：分类比较
- 饼图：占比分析
- 散点图：相关性分析
- 热力图：多维数据
- 地图：地理数据"""

    SYSTEM_ANALYSIS_AGENT = """你是一个专业的数据分析师。
你的任务是：
1. 深入分析数据
2. 发现数据中的模式和趋势
3. 提供业务洞察
4. 生成分析报告

分析维度：
- 描述性统计
- 趋势分析
- 异常检测
- 相关性分析
- 预测建议"""

    SQL_GENERATION = Template("""
用户问题：${question}

可用的数据表：
${available_tables}

请生成SQL查询语句来回答用户的问题。
要求：
1. SQL语句必须语法正确
2. 使用合适的表和字段
3. 考虑性能优化
4. 添加必要的注释

输出格式：
```sql
-- 查询说明
SELECT ...
```
""")

    CHART_RECOMMENDATION = Template("""
数据集信息：
- 行数：${row_count}
- 列数：${col_count}
- 数据类型：${data_types}
- 数据示例：
${data_sample}

用户需求：${user_request}

请推荐最合适的图表类型，并提供配置建议。

输出格式：
1. 推荐图表类型：
2. 推荐理由：
3. 图表配置：
   - X轴：
   - Y轴：
   - 其他配置：
4. 可视化建议：
""")

    DATA_ANALYSIS = Template("""
数据集概览：
${data_overview}

数据统计信息：
${statistics}

用户分析需求：${analysis_request}

请进行深入的数据分析，包括：
1. 数据特征总结
2. 关键发现
3. 趋势和模式
4. 异常和问题
5. 业务建议

输出格式请使用结构化的分析报告。
""")

    ERROR_CORRECTION = Template("""
执行出错了，错误信息：
${error_message}

原始查询：
${original_query}

请分析错误原因并提供修正后的查询。

输出修正后的查询和解释。
""")

    @classmethod
    def get_sql_prompt(cls, question: str, schema_info: Dict[str, Any]) -> str:

        schema_str = cls._format_schema(schema_info)
        tables_str = cls._format_tables(schema_info.get('tables', {}))

        system = Template(cls.SYSTEM_SQL_AGENT).substitute(schema_info=schema_str)
        user = cls.SQL_GENERATION.substitute(
            question=question,
            available_tables=tables_str
        )

        return system, user

    @classmethod
    def get_chart_prompt(
            cls,
            data_info: Dict[str, Any],
            user_request: str = "自动选择最佳图表"
    ) -> str:
        user = cls.CHART_RECOMMENDATION.substitute(
            row_count=data_info.get('row_count', 0),
            col_count=data_info.get('col_count', 0),
            data_types=data_info.get('data_types', ''),
            data_sample=data_info.get('data_sample', ''),
            user_request=user_request
        )

        return cls.SYSTEM_CHART_AGENT, user

    @classmethod
    def get_analysis_prompt(
            cls,
            data_overview: str,
            statistics: str,
            analysis_request: str
    ) -> str:
        user = cls.DATA_ANALYSIS.substitute(
            data_overview=data_overview,
            statistics=statistics,
            analysis_request=analysis_request
        )

        return cls.SYSTEM_ANALYSIS_AGENT, user

    @staticmethod
    def _format_schema(schema_info: Dict[str, Any]) -> str:
        if not schema_info:
            return "无数据库架构信息"

        lines = [f"数据库：{schema_info.get('database', 'unknown')}"]
        tables = schema_info.get('tables', {})
        lines.append(f"表数量：{len(tables)}")

        return "\n".join(lines)

    @staticmethod
    def _format_tables(tables: Dict[str, Any]) -> str:
        if not tables:
            return "无可用表"

        lines = []
        for table_name, table_info in tables.items():
            lines.append(f"\n表名：{table_name}")
            if table_info.get('comment'):
                lines.append(f"说明：{table_info['comment']}")

            columns = table_info.get('columns', [])
            if columns:
                lines.append("字段：")
                for col in columns[:10]:
                    col_line = f"  - {col['COLUMN_NAME']} ({col['DATA_TYPE']})"
                    if col.get('COLUMN_COMMENT'):
                        col_line += f" - {col['COLUMN_COMMENT']}"
                    lines.append(col_line)

                if len(columns) > 10:
                    lines.append(f"  ... 还有 {len(columns) - 10} 个字段")

        return "\n".join(lines)
