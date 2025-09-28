"""Phase 5: External Service Integration Demo

This example demonstrates the complete Phase 5 implementation including:
- LLM provider integration (OpenAI, Anthropic, Google)
- MCP server management and tool discovery
- Tool execution with security and sandboxing
- Connection pooling and health monitoring
"""

import asyncio
import json

from ai_agent.infrastructure.llm import (
    LLMProviderFactory,
    LLMProviderType,
    get_llm_provider,
    get_all_models,
    health_check_all_providers,
)
from ai_agent.infrastructure.mcp import (
    MCPServerManager,
    MCPServerType,
    MCPConnectionManager,
)
from ai_agent.infrastructure.mcp.tool_registry import (
    ToolRegistry,
    ToolCategory,
    ToolMetadata,
)
from ai_agent.infrastructure.mcp.tool_executor import (
    ToolExecutor,
    SecurityLevel,
    ExecutionEnvironment,
    SecurityPolicy,
    ExecutionContext,
)


async def demo_llm_providers():
    """Demonstrate LLM provider integration."""
    print("=== LLM Provider Integration Demo ===")

    # Register different LLM providers
    openai_id = await LLMProviderFactory.create_openai_provider(
        api_key="your-openai-key", name="OpenAI GPT-4o", default_model="gpt-4o"
    )
    print(f"✅ Registered OpenAI provider: {openai_id}")

    anthropic_id = await LLMProviderFactory.create_anthropic_provider(
        api_key="your-anthropic-key",
        name="Anthropic Claude",
        default_model="claude-3-5-sonnet-20241022",
    )
    print(f"✅ Registered Anthropic provider: {anthropic_id}")

    google_id = await LLMProviderFactory.create_google_provider(
        api_key="your-google-key",
        name="Google Gemini",
        default_model="gemini-1.5-pro",
    )
    print(f"✅ Registered Google provider: {google_id}")

    # Get all available models
    models = await get_all_models()
    print(f"📊 Total models available: {len(models)}")

    for model in models[:5]:  # Show first 5 models
        print(
            f"  - {model.name} ({model.provider}) - Functions: {model.supports_functions}"
        )

    # Health check all providers
    health_status = await health_check_all_providers()
    print(f"🏥 Provider health status: {health_status}")

    # Get best provider for a specific task
    best_provider = await get_llm_provider(provider_type=LLMProviderType.OPENAI)
    if best_provider:
        print(f"🎯 Best OpenAI provider: {best_provider.provider_type.value}")

    print()


async def demo_mcp_servers():
    """Demonstrate MCP server management."""
    print("=== MCP Server Management Demo ===")

    # Create server manager
    server_manager = MCPServerManager()
    await server_manager.start()

    # Register some example MCP servers
    file_server_id = await server_manager.register_server(
        name="File Operations Server",
        server_type=MCPServerType.PROCESS,
        endpoint="stdio://file-server",
        command=["python", "-m", "file_server"],
        description="Handles file operations",
    )
    print(f"✅ Registered file server: {file_server_id}")

    web_server_id = await server_manager.register_server(
        name="Web Scraping Server",
        server_type=MCPServerType.PROCESS,
        endpoint="stdio://web-server",
        command=["python", "-m", "web_server"],
        description="Handles web scraping operations",
    )
    print(f"✅ Registered web server: {web_server_id}")

    # Start servers
    await server_manager.start_server(file_server_id)
    await server_manager.start_server(web_server_id)

    # List all servers
    servers = await server_manager.list_servers()
    print(f"📊 Total servers: {len(servers)}")

    for server in servers:
        print(f"  - {server.name} ({server.status}) - {server.description}")

    # Health check
    health_status = await server_manager.health_check_all()
    print(f"🏥 Server health status: {health_status}")

    await server_manager.stop()
    print()


async def demo_tool_registry():
    """Demonstrate tool discovery and registration."""
    print("=== Tool Registry Demo ===")

    # Create tool registry
    tool_registry = ToolRegistry()
    await tool_registry.start()

    # Create some example tools
    from ai_agent.infrastructure.mcp.protocol import MCPTool

    file_tool = MCPTool(
        name="read_file",
        description="Read contents of a file",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"}
            },
            "required": ["path"],
        },
    )

    web_tool = MCPTool(
        name="scrape_website",
        description="Scrape content from a website",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to scrape"},
                "selector": {"type": "string", "description": "CSS selector"},
            },
            "required": ["url"],
        },
    )

    # Register tools with metadata
    file_metadata = ToolMetadata(
        category=ToolCategory.FILE_OPERATIONS,
        tags=["file", "read", "text"],
        version="1.0.0",
        author="AI Agent Team",
        rate_limit=100,
        timeout=30.0,
    )

    web_metadata = ToolMetadata(
        category=ToolCategory.WEB_SCRAPING,
        tags=["web", "scraping", "html"],
        version="1.0.0",
        author="AI Agent Team",
        rate_limit=50,
        timeout=60.0,
    )

    await tool_registry.register_tool(file_tool, "file-server", file_metadata)
    await tool_registry.register_tool(web_tool, "web-server", web_metadata)

    print("✅ Registered tools with metadata")

    # List tools by category
    file_tools = await tool_registry.list_tools(category=ToolCategory.FILE_OPERATIONS)
    print(f"📁 File operation tools: {len(file_tools)}")

    web_tools = await tool_registry.list_tools(category=ToolCategory.WEB_SCRAPING)
    print(f"🌐 Web scraping tools: {len(web_tools)}")

    # Search tools
    search_results = await tool_registry.search_tools("file", limit=5)
    print(f"🔍 Search results for 'file': {len(search_results)}")

    for tool in search_results:
        print(f"  - {tool.name}: {tool.tool.description}")

    # Get tool statistics
    stats = await tool_registry.get_tool_stats()
    print(f"📊 Tool registry stats: {json.dumps(stats, indent=2)}")

    await tool_registry.stop()
    print()


async def demo_tool_execution():
    """Demonstrate tool execution with security."""
    print("=== Tool Execution Demo ===")

    # Create tool executor
    tool_executor = ToolExecutor()

    # Create security policies
    low_security = SecurityPolicy(
        level=SecurityLevel.LOW,
        environment=ExecutionEnvironment.HOST,
        max_execution_time=60.0,
    )

    high_security = SecurityPolicy(
        level=SecurityLevel.HIGH,
        environment=ExecutionEnvironment.SANDBOX,
        max_execution_time=30.0,
        network_access=False,
        file_system_access=True,
    )

    # Create execution contexts
    low_context = ExecutionContext(
        user_id="demo_user", session_id="demo_session", security_policy=low_security
    )

    high_context = ExecutionContext(
        user_id="demo_user", session_id="demo_session", security_policy=high_security
    )

    # Create a mock tool for demonstration
    from ai_agent.infrastructure.mcp.tool_registry import RegisteredTool
    from ai_agent.infrastructure.mcp.protocol import MCPTool

    mock_tool = RegisteredTool(
        name="demo_tool",
        server_id="demo_server",
        tool=MCPTool(
            name="demo_tool",
            description="A demonstration tool",
            input_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Message to process"}
                },
                "required": ["message"],
            },
        ),
        metadata=ToolMetadata(category=ToolCategory.GENERAL, tags=["demo", "test"]),
    )

    # Execute tool with low security
    print("🔓 Executing with low security...")
    result = await tool_executor.execute_tool(
        mock_tool, {"message": "Hello, world!"}, low_context
    )
    print(f"  Result: {result.success} - {result.result}")

    # Execute tool with high security
    print("🔒 Executing with high security...")
    result = await tool_executor.execute_tool(
        mock_tool, {"message": "Hello, world!"}, high_context
    )
    print(f"  Result: {result.success} - {result.result}")

    # Test security validation
    print("🛡️ Testing security validation...")
    dangerous_tool = RegisteredTool(
        name="dangerous_tool",
        server_id="demo_server",
        tool=MCPTool(
            name="dangerous_tool",
            description="A potentially dangerous tool",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to execute"}
                },
                "required": ["command"],
            },
        ),
        metadata=ToolMetadata(category=ToolCategory.SYSTEM),
    )

    # Try to execute dangerous command
    result = await tool_executor.execute_tool(
        dangerous_tool, {"command": "rm -rf /"}, high_context
    )
    print(f"  Dangerous command blocked: {not result.success}")
    print(f"  Error: {result.error}")

    # Get execution statistics
    stats = await tool_executor.get_execution_stats()
    print(f"📊 Execution stats: {json.dumps(stats, indent=2)}")

    print()


async def demo_integration():
    """Demonstrate complete integration."""
    print("=== Complete Integration Demo ===")

    # Initialize all components
    server_manager = MCPServerManager()
    connection_manager = MCPConnectionManager()
    tool_registry = ToolRegistry()
    tool_executor = ToolExecutor()

    # Set up dependencies
    tool_registry.set_server_manager(server_manager)
    tool_registry.set_connection_manager(connection_manager)

    # Start all components
    await server_manager.start()
    await connection_manager.start()
    await tool_registry.start()

    print("✅ All components started")

    # Register LLM providers
    await LLMProviderFactory.create_openai_provider(
        api_key="demo-key", name="Demo OpenAI"
    )

    # Register MCP servers
    await server_manager.register_server(
        name="Demo Server",
        server_type=MCPServerType.PROCESS,
        endpoint="stdio://demo",
        command=["echo", "demo"],
    )

    # Discover tools
    tools = await tool_registry.discover_tools()
    print(f"🔍 Discovered {len(tools)} tools")

    # Get execution statistics
    execution_stats = await tool_executor.get_execution_stats()
    tool_stats = await tool_registry.get_tool_stats()

    print("📊 Integration stats:")
    print(f"  - Tools: {tool_stats['total_tools']}")
    print(f"  - Executions: {execution_stats['total_executions']}")

    # Clean up
    await tool_registry.stop()
    await connection_manager.stop()
    await server_manager.stop()

    print("✅ Integration demo completed")


async def main():
    """Run all demonstrations."""
    print("🚀 Phase 5: External Service Integration Demo")
    print("=" * 50)

    try:
        await demo_llm_providers()
        await demo_mcp_servers()
        await demo_tool_registry()
        await demo_tool_execution()
        await demo_integration()

        print("🎉 All demos completed successfully!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
