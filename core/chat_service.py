import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

import pandas as pd

from ai import LLMClient
from ai.agents import SQLAgent, ChartAgent, AnalysisAgent, MCPAgent
from utils import logger
from .cache_service import CacheService
from .data_service import DataService


class ChatService:

    def __init__(self):
        self.llm_client = LLMClient()
        self.sql_agent = SQLAgent(self.llm_client)
        self.chart_agent = ChartAgent(self.llm_client)
        self.analysis_agent = AnalysisAgent(self.llm_client)
        self.mcp_agent = MCPAgent(self.llm_client)
        self.data_service = DataService()
        self.cache_service = CacheService()
        self.conversations = {}
        logger.info("Initialized Chat Service")

    def create_conversation(self, user_id: Optional[str] = None) -> str:
        conversation_id = str(uuid.uuid4())

        self.conversations[conversation_id] = {
            'id': conversation_id,
            'user_id': user_id,
            'messages': [],
            'context': {},
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }

        logger.info(f"Created conversation: {conversation_id}")
        return conversation_id

    def add_message(
            self,
            conversation_id: str,
            role: str,
            content: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")

        message = {
            'id': str(uuid.uuid4()),
            'role': role,
            'content': content,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }

        self.conversations[conversation_id]['messages'].append(message)
        self.conversations[conversation_id]['updated_at'] = datetime.now().isoformat()

        return message

    async def process_message(
            self,
            conversation_id: str,
            user_message: str
    ) -> Dict[str, Any]:
        try:
            self.add_message(conversation_id, 'user', user_message)

            intent = await self._determine_intent(user_message)

            if intent['type'] == 'sql_query':
                response = await self._handle_sql_query(conversation_id, user_message, intent)
            elif intent['type'] == 'data_analysis':
                response = await self._handle_data_analysis(conversation_id, user_message, intent)
            elif intent['type'] == 'visualization':
                response = await self._handle_visualization(conversation_id, user_message, intent)
            elif intent['type'] == 'general_chat':
                response = await self._handle_general_chat(conversation_id, user_message)
            else:
                response = await self._handle_unknown_intent(conversation_id, user_message)

            self.add_message(
                conversation_id,
                'assistant',
                response['content'],
                response.get('metadata')
            )

            return response

        except Exception as e:
            logger.error(f"Failed to process message: {str(e)}")
            error_response = {
                'content': f"抱歉，处理您的请求时出现错误：{str(e)}",
                'error': True,
                'error_details': str(e)
            }

            self.add_message(conversation_id, 'assistant', error_response['content'])
            return error_response

    async def _determine_intent(self, message: str) -> Dict[str, Any]:
        prompt = f"""
分析用户消息并确定意图类型。

用户消息：{message}

可能的意图类型：
1. sql_query - 需要查询数据库
2. data_analysis - 需要分析数据
3. visualization - 需要创建图表
4. general_chat - 一般对话

请返回最匹配的意图类型和置信度（0-1）。
格式：intent_type|confidence|keywords
"""

        response = self.llm_client.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=100
        )

        parts = response.strip().split('|')
        intent = {
            'type': 'general_chat',
            'confidence': 0.5,
            'keywords': []
        }

        if len(parts) >= 2:
            intent['type'] = parts[0].strip().lower()
            try:
                intent['confidence'] = float(parts[1])
            except:
                pass

        if len(parts) >= 3:
            intent['keywords'] = [k.strip() for k in parts[2].split(',')]

        sql_keywords = ['查询', '数据', '表', 'select', 'from', '统计', '多少', '哪些']
        if any(keyword in message.lower() for keyword in sql_keywords):
            intent['type'] = 'sql_query'
            intent['confidence'] = max(intent['confidence'], 0.7)

        analysis_keywords = ['分析', '趋势', '模式', '洞察', '相关性', '异常']
        if any(keyword in message.lower() for keyword in analysis_keywords):
            intent['type'] = 'data_analysis'
            intent['confidence'] = max(intent['confidence'], 0.7)

        viz_keywords = ['图表', '可视化', '画图', '展示', '图形', 'chart', 'plot']
        if any(keyword in message.lower() for keyword in viz_keywords):
            intent['type'] = 'visualization'
            intent['confidence'] = max(intent['confidence'], 0.7)

        logger.info(f"Determined intent: {intent}")
        return intent

    async def _handle_sql_query(
            self,
            conversation_id: str,
            message: str,
            intent: Dict[str, Any]
    ) -> Dict[str, Any]:

        schema = await self.data_service.get_schema()

        sql_query, explanation = self.sql_agent.generate_sql(
            question=message,
            schema_info=schema
        )

        result = await self.data_service.execute_query(sql_query)

        self.conversations[conversation_id]['context'].update({
            'last_query_result': result.get('data', []),
            'last_sql_query': sql_query,
            'last_query_time': datetime.now().isoformat(),
            'last_query_row_count': result.get('row_count', 0)
        })

        response_content = f"""
根据您的问题，我生成了以下SQL查询：

```sql
{sql_query}
```

查询结果：
- 返回 {result.get('row_count', 0)} 条数据

{self._format_query_results(result.get('data', []))}
"""

        return {
            'content': response_content,
            'metadata': {
                'intent': 'sql_query',
                'sql': sql_query,
                'result': result
            }
        }

    async def _handle_data_analysis(
            self,
            conversation_id: str,
            message: str,
            intent: Dict[str, Any]
    ) -> Dict[str, Any]:

        context = self.conversations[conversation_id].get('context', {})

        if 'last_query_result' in context and context['last_query_result']:
            data = context['last_query_result']

            if not data:
                messages = self.conversations[conversation_id].get('messages', [])
                for msg in reversed(messages):
                    if (msg.get('metadata', {}).get('intent') == 'sql_query' and
                            msg.get('metadata', {}).get('result', {}).get('data')):
                        data = msg['metadata']['result']['data']
                        context['last_query_result'] = data
                        logger.info("Found data in message history")
                        break
        else:
            return {
                'content': "请先提供要分析的数据，或执行一个查询来获取数据。",
                'metadata': {'intent': 'data_analysis', 'needs_data': True}
            }

        try:

            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
            else:
                return {
                    'content': "数据格式不正确或为空，无法进行分析。",
                    'metadata': {'intent': 'data_analysis', 'error': 'invalid_data'}
                }

            analysis_results = self.analysis_agent.analyze_data(df, message)

            report = self.analysis_agent.generate_report(analysis_results, 'markdown')

            return {
                'content': report,
                'metadata': {
                    'intent': 'data_analysis',
                    'analysis_results': analysis_results
                }
            }
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return {
                'content': f"分析数据时出现错误：{str(e)}",
                'metadata': {'intent': 'data_analysis', 'error': str(e)}
            }

    async def _handle_visualization(
            self,
            conversation_id: str,
            message: str,
            intent: Dict[str, Any]
    ) -> Dict[str, Any]:

        context = self.conversations[conversation_id].get('context', {})

        if 'last_query_result' in context and context['last_query_result']:
            data = context['last_query_result']

            if not data:
                messages = self.conversations[conversation_id].get('messages', [])
                for msg in reversed(messages):
                    if (msg.get('metadata', {}).get('intent') == 'sql_query' and
                            msg.get('metadata', {}).get('result', {}).get('data')):
                        data = msg['metadata']['result']['data']
                        context['last_query_result'] = data
                        break
        else:
            return {
                'content': "请先提供要可视化的数据，或执行一个查询来获取数据。",
                'metadata': {'intent': 'data_analysis', 'needs_data': True}
            }

        try:
            df = pd.DataFrame(data)

            recommendation = self.chart_agent.recommend_chart(df, message)
            # chart_config = self.chart_agent.generate_chart_config(
            #     df,
            #     recommendation['chart_type']
            # )

            chart_result = self.chart_agent.generate_chart(
                df,
                recommendation['chart_type'],
                user_request=message
            )
            chart_config = chart_result['config']

            response_content = f"""
            根据您的数据，我推荐使用 **{recommendation['chart_type']}** 图表。

            推荐理由：{recommendation.get('reason', '最适合展示当前数据类型')}

            图表配置已生成，您可以使用以下配置创建图表：
            - 图表类型：{chart_config['type']}
            - X轴：{chart_config.get('x', 'N/A')}
            - Y轴：{chart_config.get('y', 'N/A')}

            建议：
            {chr(10).join('- ' + s for s in recommendation.get('suggestions', []))}
            """

            # return {
            #     'content': response_content,
            #     'metadata': {
            #         'intent': 'visualization',
            #         'chart_config': chart_config,
            #         'recommendation': recommendation
            #     }
            # }

            return {
                'content': response_content,
                'metadata': {
                    'intent': 'visualization',
                    'chart_data': chart_result['data'],
                    'chart_config': chart_result['config'],
                    'chart_type': recommendation['chart_type'],
                    'recommendation': recommendation,
                    'has_chart': True
                }
            }

        except Exception as e:
            logger.error(f"Visualization error: {str(e)}")
            return {
                'content': f"生成图表时出现错误：{str(e)}",
                'metadata': {'intent': 'visualization', 'error': str(e)}
            }

    async def _handle_general_chat(
            self,
            conversation_id: str,
            message: str
    ) -> Dict[str, Any]:

        messages = self.conversations[conversation_id]['messages']

        history = []
        for msg in messages[-10:]:
            history.append({
                'role': msg['role'],
                'content': msg['content']
            })

        response = self.llm_client.chat(
            messages=history,
            system_prompt="你是ChatBI数据智能助手，帮助用户进行数据查询、分析和可视化。"
        )

        return {
            'content': response,
            'metadata': {'intent': 'general_chat'}
        }

    async def _handle_unknown_intent(
            self,
            conversation_id: str,
            message: str
    ) -> Dict[str, Any]:
        return {
            'content': """我不太确定您的需求。我可以帮助您：
1. 查询数据库（例如："查询最近7天的订单"）
2. 分析数据（例如："分析销售趋势"）  
3. 创建图表（例如："生成销售额柱状图"）

请告诉我您想做什么？""",
            'metadata': {'intent': 'unknown'}
        }

    def _format_query_results(self, data: List[Dict], limit: int = 5) -> str:
        if not data:
            return "无数据返回"

        df = pd.DataFrame(data[:limit])

        result = "```\n"
        result += df.to_string()
        result += "\n```"

        if len(data) > limit:
            result += f"\n\n（仅显示前{limit}条，共{len(data)}条数据）"

        return result

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        return self.conversations.get(conversation_id)

    def list_conversations(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        conversations = list(self.conversations.values())

        if user_id:
            conversations = [c for c in conversations if c.get('user_id') == user_id]

        return conversations

    def delete_conversation(self, conversation_id: str) -> bool:
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Deleted conversation: {conversation_id}")
            return True
        return False

    def update_context(self, conversation_id: str, context_update: Dict[str, Any]):
        if conversation_id in self.conversations:
            self.conversations[conversation_id]['context'].update(context_update)
            self.conversations[conversation_id]['updated_at'] = datetime.now().isoformat()
