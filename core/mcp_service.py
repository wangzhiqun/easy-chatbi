import asyncio
from typing import Dict, Any, List, Optional

from mcpv1 import MCPServer, MCPClient, MCPTools, MCPResources, MCPPrompts
from utils import logger, MCPError


class MCPService:

    def __init__(self):
        self.server = MCPServer()
        self.client = MCPClient()
        self.tools = MCPTools()
        self.resources = MCPResources()
        self.prompts = MCPPrompts()
        self.active_sessions = {}
        logger.info("Initialized MCP Service")

    async def start_server(self, transport: str = "stdio"):
        try:
            await self.server.start(transport)
            logger.info(f"MCP server started with transport: {transport}")
        except Exception as e:
            logger.error(f"Failed to start MCP server: {str(e)}")
            raise MCPError(f"Failed to start MCP server: {str(e)}")

    async def connect_to_server(
            self,
            server_command: List[str],
            session_id: Optional[str] = None
    ) -> str:
        try:
            await self.client.connect(server_command)

            session_id = session_id or f"session_{len(self.active_sessions)}"
            self.active_sessions[session_id] = {
                'client': self.client,
                'server_command': server_command,
                'connected_at': asyncio.get_event_loop().time()
            }

            logger.info(f"Connected to MCP server: {session_id}")
            return session_id

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {str(e)}")
            raise MCPError(f"Failed to connect to MCP server: {str(e)}")

    async def execute_tool(
            self,
            tool_name: str,
            arguments: Optional[Dict[str, Any]] = None,
            session_id: Optional[str] = None
    ) -> Any:
        try:
            if session_id and session_id in self.active_sessions:
                client = self.active_sessions[session_id]['client']
                result = await client.call_tool(tool_name, arguments)
            else:
                if tool_name == 'query_database':
                    result = await self.tools.query_database(
                        arguments.get('query'),
                        arguments.get('database')
                    )
                elif tool_name == 'generate_sql':
                    result = await self.tools.generate_sql(arguments.get('question'))
                elif tool_name == 'create_chart':
                    result = await self.tools.create_chart(
                        arguments.get('data'),
                        arguments.get('chart_type', 'auto'),
                        arguments.get('options')
                    )
                elif tool_name == 'analyze_data':
                    result = await self.tools.analyze_data(
                        arguments.get('data'),
                        arguments.get('analysis_type', 'comprehensive')
                    )
                elif tool_name == 'validate_sql':
                    result = await self.tools.validate_sql(arguments.get('sql_query'))
                elif tool_name == 'get_table_info':
                    result = await self.tools.get_table_info(arguments.get('table_name'))
                else:
                    raise ValueError(f"Unknown tool: {tool_name}")

            logger.info(f"Executed tool: {tool_name}")
            return result

        except Exception as e:
            logger.error(f"Tool execution failed: {str(e)}")
            raise MCPError(f"Tool execution failed: {str(e)}")

    async def get_resource(
            self,
            resource_uri: str,
            session_id: Optional[str] = None
    ) -> Any:
        try:
            if session_id and session_id in self.active_sessions:
                client = self.active_sessions[session_id]['client']
                result = await client.read_resource(resource_uri)
            else:
                if resource_uri == 'db://schema':
                    result = await self.resources.get_database_schema()
                elif resource_uri == 'db://tables':
                    result = await self.resources.get_table_list()
                elif resource_uri == 'history://queries':
                    result = await self.resources.get_query_history()
                elif resource_uri == 'templates://analysis':
                    result = await self.resources.get_analysis_templates()
                elif resource_uri == 'templates://charts':
                    result = await self.resources.get_chart_templates()
                elif resource_uri == 'metadata://dictionary':
                    result = await self.resources.get_data_dictionary()
                elif resource_uri == 'cache://results':
                    result = await self.resources.get_cached_results()
                else:
                    raise ValueError(f"Unknown resource: {resource_uri}")

            logger.info(f"Retrieved resource: {resource_uri}")
            return result

        except Exception as e:
            logger.error(f"Failed to get resource: {str(e)}")
            raise MCPError(f"Failed to get resource: {str(e)}")

    async def execute_prompt(
            self,
            prompt_name: str,
            arguments: Dict[str, str],
            session_id: Optional[str] = None
    ) -> str:
        try:
            if session_id and session_id in self.active_sessions:
                client = self.active_sessions[session_id]['client']
                result = await client.execute_prompt(prompt_name, arguments)
            else:
                prompt_data = self.prompts.get_prompt(prompt_name, arguments)
                result = f"{prompt_data['system']}\n\n{prompt_data['user']}"

            logger.info(f"Executed prompt: {prompt_name}")
            return result

        except Exception as e:
            logger.error(f"Prompt execution failed: {str(e)}")
            raise MCPError(f"Prompt execution failed: {str(e)}")

    async def list_available_tools(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            if session_id and session_id in self.active_sessions:
                client = self.active_sessions[session_id]['client']
                tools = await client.list_tools()
            else:
                tools = self.tools.get_tool_definitions()

            return tools

        except Exception as e:
            logger.error(f"Failed to list tools: {str(e)}")
            raise MCPError(f"Failed to list tools: {str(e)}")

    async def list_available_resources(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            if session_id and session_id in self.active_sessions:
                client = self.active_sessions[session_id]['client']
                resources = await client.list_resources()
            else:
                resources = self.resources.get_resource_definitions()

            return resources

        except Exception as e:
            logger.error(f"Failed to list resources: {str(e)}")
            raise MCPError(f"Failed to list resources: {str(e)}")

    async def list_available_prompts(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            if session_id and session_id in self.active_sessions:
                client = self.active_sessions[session_id]['client']
                prompts = await client.list_prompts()
            else:
                prompts = self.prompts.get_prompt_definitions()

            return prompts

        except Exception as e:
            logger.error(f"Failed to list prompts: {str(e)}")
            raise MCPError(f"Failed to list prompts: {str(e)}")

    async def disconnect_session(self, session_id: str) -> bool:
        try:
            if session_id in self.active_sessions:
                client = self.active_sessions[session_id]['client']
                await client.disconnect()
                del self.active_sessions[session_id]
                logger.info(f"Disconnected session: {session_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to disconnect session: {str(e)}")
            return False

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            return {
                'session_id': session_id,
                'server_command': session['server_command'],
                'connected_at': session['connected_at'],
                'is_active': True
            }
        return None

    def list_sessions(self) -> List[str]:
        return list(self.active_sessions.keys())
