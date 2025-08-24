from typing import Dict, List, Any, Optional


class MCPPrompts:
    SQL_GENERATION_SYSTEM = """你是一个专业的SQL查询专家。你的任务是：
1. 理解用户的自然语言查询需求
2. 基于提供的数据库架构生成准确的SQL查询
3. 确保查询安全高效
4. 只生成SELECT查询，不修改数据

重要规则：
- 使用标准SQL语法
- 添加适当的WHERE、JOIN、GROUP BY子句
- 考虑性能优化
- 避免笛卡尔积"""

    SQL_GENERATION_USER = """数据库架构：
{schema}

用户问题：{question}

请生成SQL查询并简要说明查询逻辑。"""

    DATA_ANALYSIS_SYSTEM = """你是一个专业的数据分析师。你的任务是：
1. 深入分析提供的数据
2. 发现数据中的模式、趋势和异常
3. 提供有价值的业务洞察
4. 给出可操作的建议

分析维度：
- 描述性统计
- 趋势分析
- 相关性分析
- 异常检测"""

    DATA_ANALYSIS_USER = """数据概览：
{data_overview}

分析目标：{analysis_goal}

请进行全面分析并提供洞察。"""

    CHART_RECOMMENDATION_SYSTEM = """你是一个数据可视化专家。你的任务是：
1. 根据数据特征推荐最合适的图表类型
2. 解释选择理由
3. 提供图表配置建议
4. 考虑用户体验和美观性

可用图表类型：
- 折线图：时间序列、趋势
- 柱状图：分类比较
- 饼图：占比分析
- 散点图：相关性
- 热力图：多维数据
- 箱线图：分布分析"""

    CHART_RECOMMENDATION_USER = """数据类型：{data_types}
数据示例：
{data_sample}

可视化需求：{visualization_need}

请推荐最佳图表类型和配置。"""

    ERROR_DIAGNOSIS_SYSTEM = """你是一个技术支持专家。你的任务是：
1. 分析错误信息
2. 诊断问题原因
3. 提供解决方案
4. 给出预防建议"""

    ERROR_DIAGNOSIS_USER = """错误信息：
{error_message}

上下文：
{context}

请诊断问题并提供解决方案。"""

    REPORT_GENERATION_SYSTEM = """你是一个报告撰写专家。你的任务是：
1. 组织和结构化信息
2. 突出关键发现
3. 使用清晰简洁的语言
4. 提供执行摘要

报告结构：
- 执行摘要
- 关键发现
- 详细分析
- 建议和下一步"""

    REPORT_GENERATION_USER = """分析结果：
{analysis_results}

报告类型：{report_type}

请生成一份结构化的分析报告。"""

    @classmethod
    def get_prompt_definitions(cls) -> List[Dict[str, Any]]:
        return [
            {
                'name': 'sql_generation',
                'description': '将自然语言转换为SQL查询',
                'arguments': [
                    {'name': 'question', 'required': True, 'description': '用户的查询问题'},
                    {'name': 'schema', 'required': False, 'description': '数据库架构信息'}
                ]
            },
            {
                'name': 'data_analysis',
                'description': '对数据进行深入分析',
                'arguments': [
                    {'name': 'data_overview', 'required': True, 'description': '数据概览'},
                    {'name': 'analysis_goal', 'required': False, 'description': '分析目标'}
                ]
            },
            {
                'name': 'chart_recommendation',
                'description': '推荐合适的图表类型',
                'arguments': [
                    {'name': 'data_types', 'required': True, 'description': '数据类型信息'},
                    {'name': 'data_sample', 'required': True, 'description': '数据样本'},
                    {'name': 'visualization_need', 'required': False, 'description': '可视化需求'}
                ]
            },
            {
                'name': 'error_diagnosis',
                'description': '诊断和解决错误',
                'arguments': [
                    {'name': 'error_message', 'required': True, 'description': '错误信息'},
                    {'name': 'context', 'required': False, 'description': '错误上下文'}
                ]
            },
            {
                'name': 'report_generation',
                'description': '生成分析报告',
                'arguments': [
                    {'name': 'analysis_results', 'required': True, 'description': '分析结果'},
                    {'name': 'report_type', 'required': False, 'description': '报告类型'}
                ]
            }
        ]

    @classmethod
    def get_prompt(cls, name: str, arguments: Dict[str, str]) -> Dict[str, str]:
        prompts = {
            'sql_generation': {
                'system': cls.SQL_GENERATION_SYSTEM,
                'user': cls.SQL_GENERATION_USER.format(
                    schema=arguments.get('schema', 'No schema provided'),
                    question=arguments.get('question', '')
                )
            },
            'data_analysis': {
                'system': cls.DATA_ANALYSIS_SYSTEM,
                'user': cls.DATA_ANALYSIS_USER.format(
                    data_overview=arguments.get('data_overview', ''),
                    analysis_goal=arguments.get('analysis_goal', '全面分析')
                )
            },
            'chart_recommendation': {
                'system': cls.CHART_RECOMMENDATION_SYSTEM,
                'user': cls.CHART_RECOMMENDATION_USER.format(
                    data_types=arguments.get('data_types', ''),
                    data_sample=arguments.get('data_sample', ''),
                    visualization_need=arguments.get('visualization_need', '自动选择')
                )
            },
            'error_diagnosis': {
                'system': cls.ERROR_DIAGNOSIS_SYSTEM,
                'user': cls.ERROR_DIAGNOSIS_USER.format(
                    error_message=arguments.get('error_message', ''),
                    context=arguments.get('context', 'No context provided')
                )
            },
            'report_generation': {
                'system': cls.REPORT_GENERATION_SYSTEM,
                'user': cls.REPORT_GENERATION_USER.format(
                    analysis_results=arguments.get('analysis_results', ''),
                    report_type=arguments.get('report_type', '综合报告')
                )
            }
        }

        return prompts.get(name, {'system': '', 'user': f'Unknown prompt: {name}'})

    @classmethod
    def create_custom_prompt(
            cls,
            template: str,
            variables: Dict[str, str],
            system_context: Optional[str] = None
    ) -> Dict[str, str]:
        try:
            user_prompt = template
            for key, value in variables.items():
                placeholder = f'{{{key}}}'
                user_prompt = user_prompt.replace(placeholder, str(value))

            return {
                'system': system_context or "You are a helpful assistant.",
                'user': user_prompt
            }
        except Exception as e:
            return {
                'system': '',
                'user': f'Error creating prompt: {str(e)}'
            }

    @classmethod
    def chain_prompts(cls, prompts: List[Dict[str, str]]) -> List[Dict[str, str]]:
        messages = []

        for i, prompt in enumerate(prompts):
            if i == 0:
                messages.append({'role': 'system', 'content': prompt.get('system', '')})

            messages.append({'role': 'user', 'content': prompt.get('user', '')})

            if i < len(prompts) - 1:
                messages.append({'role': 'assistant', 'content': '[Previous response]'})

        return messages
