import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from mcp import ClientSession, StdioServerParameters
from mcp import types as MCPTypes
from mcp.client.stdio import stdio_client

from utils import logger
from utils.exceptions import MCPAgentError
from ..llm_client import LLMClient


@dataclass
class MCPAgentConfig:
    server_command: List[str]
    server_args: List[str] = None
    timeout: int = 30
    max_retries: int = 3
    enable_tools: bool = True
    enable_resources: bool = True


class MCPToolWrapper(BaseTool):
    name: str
    description: str

    def __init__(self, mcp_tool: MCPTypes.Tool, client_session: ClientSession):
        super().__init__(
            name=mcp_tool.name,
            description=mcp_tool.description or f"MCP tool: {mcp_tool.name}"
        )

        self._mcp_tool = mcp_tool
        self._client_session = client_session

    def _run(self, **kwargs) -> str:
        return asyncio.run(self._arun(**kwargs))

    async def _arun(self, **kwargs) -> str:
        try:
            result = await self._client_session.call_tool(
                self.name,
                kwargs
            )

            if hasattr(result, 'isError') and result.isError:
                raise MCPAgentError(f"MCP tool error: {result.content}")

            if hasattr(result, 'content') and result.content:
                if isinstance(result.content, list):
                    return '\n'.join(str(item.text) for item in result.content if hasattr(item, 'text'))
                elif hasattr(result.content[0], 'text'):
                    return result.content[0].text
                return str(result.content)

            return "Tool executed successfully"

        except Exception as e:
            logger.error(f"Error executing MCP tool {self.name}: {e}")
            raise MCPAgentError(f"Failed to execute tool: {e}")


class MCPResourceManager:

    def __init__(self, client_session: ClientSession):
        self.client_session = client_session
        self._resources_cache: Dict[str, Any] = {}

    async def list_resources(self) -> List[MCPTypes.Resource]:
        try:
            result = await self.client_session.list_resources()
            return result.resources if hasattr(result, 'resources') else []
        except Exception as e:
            logger.error(f"Error listing MCP resources: {e}")
            raise MCPAgentError(f"Failed to list resources: {e}")

    async def read_resource(self, uri: str, use_cache: bool = True) -> str:
        if use_cache and uri in self._resources_cache:
            return self._resources_cache[uri]

        try:
            result = await self.client_session.read_resource(uri)

            content = ""
            if hasattr(result, 'contents') and result.contents:
                if isinstance(result.contents, list):
                    content = '\n'.join(str(item.text) for item in result.contents if hasattr(item, 'text'))
                elif hasattr(result.contents[0], 'text'):
                    content = result.contents[0].text
                else:
                    content = str(result.contents)

            if use_cache:
                self._resources_cache[uri] = content

            return content

        except Exception as e:
            logger.error(f"Error reading MCP resource {uri}: {e}")
            raise MCPAgentError(f"Failed to read resource: {e}")

    def clear_cache(self):
        self._resources_cache.clear()


class MCPAgent:

    def __init__(
            self,
            config: MCPAgentConfig,
            llm_client: Optional[LLMClient] = None
    ):
        self.config = config
        self.llm_client = llm_client or LLMClient()
        self.client_session: Optional[ClientSession] = None
        self.resource_manager: Optional[MCPResourceManager] = None
        self.tools: List[MCPToolWrapper] = []
        self.agent_executor: Optional[AgentExecutor] = None
        self._is_initialized = False

    async def initialize(self) -> bool:
        if self._is_initialized:
            return True

        try:
            server_params = StdioServerParameters(
                command=self.config.server_command[0],
                args=self.config.server_command[1:] + (self.config.server_args or [])
            )

            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as s:
                    await s.initialize()

            # transport = await stdio_client(server_params)
            # read_stream, write_stream = transport
            #
            # self.client_session = ClientSession(read_stream, write_stream)
            # await self.client_session.initialize()

            self.resource_manager = MCPResourceManager(self.client_session)

            if self.config.enable_tools:
                await self._load_tools()

            await self._setup_agent()

            self._is_initialized = True
            logger.info("MCP Agent initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize MCP Agent: {e}")
            raise MCPAgentError(f"Initialization failed: {e}")

    async def _load_tools(self):
        try:
            tools_result = await self.client_session.list_tools()
            mcp_tools = tools_result.tools
            logger.info(f"MCP tools result: {tools_result.tools}")
            self.tools = [
                MCPToolWrapper(tool, self.client_session)
                for tool in mcp_tools
            ]

            logger.info(f"Loaded {len(self.tools)} MCP tools")

        except Exception as e:
            logger.error(f"Error loading MCP tools: {e}")
            raise MCPAgentError(f"Failed to load tools: {e}")

    async def _setup_agent(self):
        try:
            system_prompt = """你是一个专业的数据智能分析助手，具备以下能力：

1. 数据查询与分析：能够理解用户的数据需求，生成准确的SQL查询
2. 图表可视化：根据数据特征选择合适的图表类型进行可视化
3. 智能洞察：从数据中发现趋势、异常和业务见解
4. 工具集成：熟练使用各种MCP工具进行数据处理

请始终：
- 理解用户的业务意图
- 提供准确可靠的数据分析
- 用清晰简洁的语言解释结果
- 在必要时使用可视化图表
- 确保数据安全和隐私保护

当前可用的工具：{tool_names}
"""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad")
            ])

            llm = self.llm_client.get_llm()

            agent = create_openai_tools_agent(
                llm=llm,
                tools=self.tools,
                prompt=prompt
            )

            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=10,
                early_stopping_method="generate"
            )

            logger.info("LangChain agent setup completed")

        except Exception as e:
            logger.error(f"Error setting up agent: {e}")
            raise MCPAgentError(f"Failed to setup agent: {e}")

    async def chat(
            self,
            message: str,
            chat_history: Optional[List[Dict]] = None,
            context: Optional[Dict] = None
    ) -> Dict[str, Any]:

        if not self._is_initialized:
            await self.initialize()

        try:
            agent_input = {
                "input": message,
                "tool_names": [tool.name for tool in self.tools]
            }

            if chat_history:
                formatted_history = []
                for msg in chat_history:
                    if msg.get("role") == "user":
                        formatted_history.append(HumanMessage(content=msg["content"]))
                    elif msg.get("role") == "assistant":
                        formatted_history.append(AIMessage(content=msg["content"]))

                agent_input["chat_history"] = formatted_history

            if context and self.config.enable_resources:
                await self._inject_context_resources(context)

            result = await self.agent_executor.ainvoke(agent_input)

            return {
                "response": result["output"],
                "intermediate_steps": result.get("intermediate_steps", []),
                "tools_used": [step[0].tool for step in result.get("intermediate_steps", [])],
                "success": True
            }

        except Exception as e:
            logger.error(f"Error in MCP agent chat: {e}")
            return {
                "response": f"抱歉，处理您的请求时出现错误：{str(e)}",
                "error": str(e),
                "success": False
            }

    async def _inject_context_resources(self, context: Dict):
        try:
            if "resource_uris" in context:
                resource_contents = []
                for uri in context["resource_uris"]:
                    content = await self.resource_manager.read_resource(uri)
                    resource_contents.append(f"Resource {uri}:\n{content}")

                if resource_contents:
                    context["resource_context"] = "\n\n".join(resource_contents)

        except Exception as e:
            logger.warning(f"Failed to inject context resources: {e}")

    async def get_available_tools(self) -> List[Dict[str, str]]:
        if not self._is_initialized:
            await self.initialize()

        return [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in self.tools
        ]

    async def get_available_resources(self) -> List[Dict[str, str]]:
        if not self._is_initialized:
            await self.initialize()

        if not self.resource_manager:
            return []

        try:
            resources = await self.resource_manager.list_resources()
            return [
                {
                    "uri": resource.uri,
                    "name": resource.name or resource.uri,
                    "description": resource.description or "MCP Resource"
                }
                for resource in resources
            ]
        except Exception as e:
            logger.error(f"Error getting resources: {e}")
            return []

    async def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        if not self._is_initialized:
            await self.initialize()

        try:
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if not tool:
                raise MCPAgentError(f"Tool {tool_name} not found")

            result = await tool._arun(**kwargs)
            return {
                "result": result,
                "success": True
            }

        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return {
                "error": str(e),
                "success": False
            }

    async def close(self):
        try:
            if self.resource_manager:
                self.resource_manager.clear_cache()

            if self.client_session:
                pass

            self._is_initialized = False
            logger.info("MCP Agent closed successfully")

        except Exception as e:
            logger.error(f"Error closing MCP Agent: {e}")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class MCPAgentFactory:

    @staticmethod
    def create_data_agent(server_command: List[str]) -> MCPAgent:
        config = MCPAgentConfig(
            server_command=server_command,
            enable_tools=True,
            enable_resources=True,
            timeout=60
        )
        return MCPAgent(config)

    @staticmethod
    def create_chart_agent(server_command: List[str]) -> MCPAgent:
        config = MCPAgentConfig(
            server_command=server_command,
            enable_tools=True,
            enable_resources=False,
            timeout=30
        )
        return MCPAgent(config)

    @staticmethod
    def create_analysis_agent(server_command: List[str]) -> MCPAgent:
        config = MCPAgentConfig(
            server_command=server_command,
            enable_tools=True,
            enable_resources=True,
            timeout=90
        )
        return MCPAgent(config)


async def main():
    try:
        server_command = ["python", "-m", "mcpv1.server"]
        config = MCPAgentConfig(server_command=server_command)

        async with stdio_client(StdioServerParameters(
                command=server_command[0],
                args=server_command[1:]
        )) as transport:
            read_stream, write_stream = transport

            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                agent = MCPAgent(config)
                agent.client_session = session
                agent.resource_manager = MCPResourceManager(session)

                await agent._load_tools()

                response = await agent.chat(
                    "请帮我分析销售数据的趋势",
                    context={"resource_uris": ["file://data/sales.csv"]}
                )
                print(f"Agent Response: {response['response']}")

                tools = await agent.get_available_tools()
                print(f"Available Tools: {tools}")

    except Exception as e:
        logger.error(f"Example execution failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
