import asyncio
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from utils import logger, MCPError


class MCPClient:

    def __init__(self):
        self.session = None
        self.server_params = None
        logger.info("Initialized MCP Client")

    async def connect(self, server_command: List[str], server_args: Optional[List[str]] = None):
        try:

            self.server_params = StdioServerParameters(
                command=server_command[0],
                args=server_command[1:] if len(server_command) > 1 else (server_args or []),
                env=None
            )

            async with stdio_client(self.server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as s:
                    await asyncio.wait_for(s.initialize(), timeout=30.0)

            logger.info(f"Connected to MCP server: {server_command}")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {str(e)}")
            raise MCPError(f"MCP connection failed: {str(e)}")

    async def disconnect(self):
        if self.session:
            try:
                await self.session.close()
                self.session = None
                logger.info("Disconnected from MCP server")
            except Exception as e:
                logger.error(f"Error disconnecting from MCP server: {str(e)}")

    async def list_tools(self) -> List[Dict[str, Any]]:
        if not self.session:
            raise MCPError("Not connected to MCP server")

        try:
            result = await self.session.list_tools()
            tools = []
            for tool in result.tools:
                tools.append({
                    'name': tool.name,
                    'description': tool.description,
                    'inputSchema': tool.inputSchema
                })

            logger.info(f"Retrieved {len(tools)} tools from server")
            return tools

        except Exception as e:
            logger.error(f"Failed to list tools: {str(e)}")
            raise MCPError(f"Failed to list tools: {str(e)}")

    async def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        if not self.session:
            raise MCPError("Not connected to MCP server")

        try:
            result = await self.session.call_tool(tool_name, arguments or {})

            if result.content:
                content = []
                for item in result.content:
                    if hasattr(item, 'text'):
                        content.append(item.text)
                    elif hasattr(item, 'data'):
                        content.append(item.data)

                logger.info(f"Tool {tool_name} executed successfully")
                return content[0] if len(content) == 1 else content

            return None

        except Exception as e:
            logger.error(f"Failed to call tool {tool_name}: {str(e)}")
            raise MCPError(f"Failed to call tool: {str(e)}")

    async def list_resources(self) -> List[Dict[str, Any]]:
        if not self.session:
            raise MCPError("Not connected to MCP server")

        try:
            result = await self.session.list_resources()
            resources = []
            for resource in result.resources:
                resources.append({
                    'uri': resource.uri,
                    'name': resource.name,
                    'description': resource.description,
                    'mimeType': resource.mimeType
                })

            logger.info(f"Retrieved {len(resources)} resources from server")
            return resources

        except Exception as e:
            logger.error(f"Failed to list resources: {str(e)}")
            raise MCPError(f"Failed to list resources: {str(e)}")

    async def read_resource(self, uri: str) -> Any:
        if not self.session:
            raise MCPError("Not connected to MCP server")

        try:
            result = await self.session.read_resource(uri)

            if result.contents:
                content = []
                for item in result.contents:
                    if hasattr(item, 'text'):
                        content.append(item.text)
                    elif hasattr(item, 'data'):
                        content.append(item.data)

                logger.info(f"Resource {uri} read successfully")
                return content[0] if len(content) == 1 else content

            return None

        except Exception as e:
            logger.error(f"Failed to read resource {uri}: {str(e)}")
            raise MCPError(f"Failed to read resource: {str(e)}")

    async def list_prompts(self) -> List[Dict[str, Any]]:
        if not self.session:
            raise MCPError("Not connected to MCP server")

        try:
            result = await self.session.list_prompts()
            prompts = []
            for prompt in result.prompts:
                prompts.append({
                    'name': prompt.name,
                    'description': prompt.description,
                    'arguments': prompt.arguments if hasattr(prompt, 'arguments') else []
                })

            logger.info(f"Retrieved {len(prompts)} prompts from server")
            return prompts

        except Exception as e:
            logger.error(f"Failed to list prompts: {str(e)}")
            raise MCPError(f"Failed to list prompts: {str(e)}")

    async def get_prompt(self, prompt_name: str, arguments: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        if not self.session:
            raise MCPError("Not connected to MCP server")

        try:
            result = await self.session.get_prompt(prompt_name, arguments or {})

            messages = []
            if hasattr(result, 'messages'):
                for msg in result.messages:
                    message = {
                        'role': msg.role,
                        'content': ''
                    }
                    if hasattr(msg, 'content'):
                        if hasattr(msg.content, 'text'):
                            message['content'] = msg.content.text
                        elif isinstance(msg.content, str):
                            message['content'] = msg.content
                    messages.append(message)

            logger.info(f"Prompt {prompt_name} retrieved successfully")
            return {
                'name': prompt_name,
                'messages': messages
            }

        except Exception as e:
            logger.error(f"Failed to get prompt {prompt_name}: {str(e)}")
            raise MCPError(f"Failed to get prompt: {str(e)}")

    async def execute_prompt(
            self,
            prompt_name: str,
            arguments: Optional[Dict[str, str]] = None
    ) -> str:
        try:
            prompt_data = await self.get_prompt(prompt_name, arguments)

            formatted = []
            for msg in prompt_data.get('messages', []):
                formatted.append(f"{msg['role']}: {msg['content']}")

            return "\n\n".join(formatted)

        except Exception as e:
            logger.error(f"Failed to execute prompt: {str(e)}")
            raise MCPError(f"Failed to execute prompt: {str(e)}")


# Example usage
async def example_usage():
    """Example of using MCP Client"""
    client = MCPClient()

    try:
        # Connect to a local MCP server
        await client.connect(["python", "-m", "mcpv1.server"])

        # List available tools
        tools = await client.list_tools()
        print(f"Available tools: {tools}")

        # Call a tool
        result = await client.call_tool(
            "query_database",
            {"query": "SELECT * FROM users LIMIT 5"}
        )
        print(f"Query result: {result}")

        # List resources
        resources = await client.list_resources()
        print(f"Available resources: {resources}")

        # Read a resource
        schema = await client.read_resource("db://schema")
        print(f"Schema: {schema}")

        # Get a prompt
        prompt = await client.get_prompt(
            "sql_generation",
            {"question": "Show me all users who joined last month"}
        )
        print(f"Prompt: {prompt}")

    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(example_usage())
