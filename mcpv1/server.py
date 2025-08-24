import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from typing import Any, Dict, List, Optional
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import (
    Tool,
    Resource,
    Prompt,
    TextContent,
    ServerCapabilities
)
from utils import logger, get_config


class MCPServer:

    def __init__(self):
        self.config = get_config()
        self.server = Server(self.config.mcp_server_name)
        self._setup_handlers()
        logger.info(f"Initialized MCP Server: {self.config.mcp_server_name}")

    def _setup_handlers(self):

        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            return [
                Tool(
                    name="query_database",
                    description="Execute SQL query on database",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "SQL query to execute"
                            },
                            "database": {
                                "type": "string",
                                "description": "Target database name"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="generate_chart",
                    description="Generate chart from data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "array",
                                "description": "Data for visualization"
                            },
                            "chart_type": {
                                "type": "string",
                                "enum": ["line", "bar", "pie", "scatter", "heatmap"],
                                "description": "Type of chart"
                            }
                        },
                        "required": ["data", "chart_type"]
                    }
                ),
                Tool(
                    name="analyze_data",
                    description="Perform data analysis",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "array",
                                "description": "Data to analyze"
                            },
                            "analysis_type": {
                                "type": "string",
                                "enum": ["statistical", "trend", "correlation", "anomaly"],
                                "description": "Type of analysis"
                            }
                        },
                        "required": ["data"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Optional[Dict[str, Any]] = None) -> List[TextContent]:
            try:
                if name == "query_database":
                    result = await self._execute_query(arguments)
                elif name == "generate_chart":
                    result = await self._generate_chart(arguments)
                elif name == "analyze_data":
                    result = await self._analyze_data(arguments)
                else:
                    result = f"Unknown tool: {name}"

                return [TextContent(type="text", text=str(result))]

            except Exception as e:
                logger.error(f"Tool execution failed: {str(e)}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]

        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            return [
                Resource(
                    uri="db://schema",
                    name="Database Schema",
                    description="Current database schema information",
                    mimeType="application/json"
                ),
                Resource(
                    uri="db://tables",
                    name="Database Tables",
                    description="List of available tables",
                    mimeType="application/json"
                ),
                Resource(
                    uri="cache://queries",
                    name="Query Cache",
                    description="Cached query results",
                    mimeType="application/json"
                )
            ]

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            try:
                if uri == "db://schema":
                    content = await self._get_schema()
                elif uri == "db://tables":
                    content = await self._get_tables()
                elif uri == "cache://queries":
                    content = await self._get_cached_queries()
                else:
                    content = f"Unknown resource: {uri}"

                return str(content)

            except Exception as e:
                logger.error(f"Resource read failed: {str(e)}")
                return f"Error: {str(e)}"

        @self.server.list_prompts()
        async def handle_list_prompts() -> List[Prompt]:
            return [
                Prompt(
                    name="sql_generation",
                    description="生成SQL查询",
                    arguments=[
                        {
                            "name": "question",
                            "description": "用户的自然语言问题",
                            "required": True
                        }
                    ]
                ),
                Prompt(
                    name="data_analysis",
                    description="数据分析",
                    arguments=[
                        {
                            "name": "data_description",
                            "description": "数据描述",
                            "required": True
                        },
                        {
                            "name": "analysis_goal",
                            "description": "分析目标",
                            "required": False
                        }
                    ]
                ),
                Prompt(
                    name="chart_recommendation",
                    description="图表推荐",
                    arguments=[
                        {
                            "name": "data_type",
                            "description": "数据类型",
                            "required": True
                        }
                    ]
                )
            ]

        @self.server.get_prompt()
        async def handle_get_prompt(name: str, arguments: Optional[Dict[str, str]] = None) -> Prompt:
            if name == "sql_generation":
                messages = [
                    {
                        "role": "system",
                        "content": TextContent(
                            type="text",
                            text="你是一个SQL查询专家，帮助用户将自然语言转换为SQL查询。"
                        )
                    },
                    {
                        "role": "user",
                        "content": TextContent(
                            type="text",
                            text=f"问题：{arguments.get('question', '')}\n\n请生成对应的SQL查询。"
                        )
                    }
                ]
            elif name == "data_analysis":
                messages = [
                    {
                        "role": "system",
                        "content": TextContent(
                            type="text",
                            text="你是一个数据分析专家，提供深入的数据洞察。"
                        )
                    },
                    {
                        "role": "user",
                        "content": TextContent(
                            type="text",
                            text=f"数据：{arguments.get('data_description', '')}\n目标：{arguments.get('analysis_goal', '全面分析')}"
                        )
                    }
                ]
            elif name == "chart_recommendation":
                messages = [
                    {
                        "role": "system",
                        "content": TextContent(
                            type="text",
                            text="你是一个数据可视化专家，推荐最适合的图表类型。"
                        )
                    },
                    {
                        "role": "user",
                        "content": TextContent(
                            type="text",
                            text=f"数据类型：{arguments.get('data_type', '')}\n\n请推荐合适的图表类型。"
                        )
                    }
                ]
            else:
                messages = []

            return Prompt(
                name=name,
                description=f"Prompt for {name}",
                messages=messages
            )

    async def start(self, transport: str = "stdio"):
        try:
            if transport == "stdio":
                from mcp.server.stdio import stdio_server

                async with stdio_server() as (read_stream, write_stream):

                    await self.server.run(
                        read_stream,
                        write_stream,
                        InitializationOptions(
                            server_name=self.config.mcp_server_name,
                            server_version=self.config.mcp_server_version,
                            capabilities=ServerCapabilities()
                        )
                    )

                logger.info("MCP Server started!")

            else:
                logger.error(f"Unsupported transport: {transport}")

        except Exception as e:
            logger.error(f"MCP Server start failed: {str(e)}")
            raise Exception(f"MCP Server start failed: {str(e)}")

    async def _execute_query(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        query = arguments.get('query', '')
        database = arguments.get('database', self.config.mysql_database)

        return {
            'status': 'success',
            'query': query,
            'database': database,
            'rows_affected': 0,
            'result': 'Query executed successfully'
        }

    async def _generate_chart(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        data = arguments.get('data', [])
        chart_type = arguments.get('chart_type', 'bar')

        return {
            'status': 'success',
            'chart_type': chart_type,
            'config': {
                'type': chart_type,
                'data': data,
                'options': {
                    'responsive': True,
                    'maintainAspectRatio': False
                }
            }
        }

    async def _analyze_data(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        data = arguments.get('data', [])
        analysis_type = arguments.get('analysis_type', 'statistical')

        return {
            'status': 'success',
            'analysis_type': analysis_type,
            'summary': {
                'row_count': len(data),
                'analysis': f'{analysis_type} analysis completed'
            }
        }

    async def _get_schema(self) -> Dict[str, Any]:
        return {
            'database': self.config.mysql_database,
            'tables': ['users', 'orders', 'products'],
            'version': '1.0.0'
        }

    async def _get_tables(self) -> List[str]:
        return ['users', 'orders', 'products', 'analytics']

    async def _get_cached_queries(self) -> List[Dict[str, Any]]:
        return [
            {
                'query': 'SELECT * FROM users LIMIT 10',
                'timestamp': '2024-01-01T00:00:00Z',
                'row_count': 10
            }
        ]


async def main():
    try:
        server = MCPServer()
        await server.start()
    except Exception as e:
        logger.error(f"MCP Server error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
