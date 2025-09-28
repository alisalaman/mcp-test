-- Initial database schema for AI Agent application
-- This script creates all the necessary tables, indexes, and triggers

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255),
    title VARCHAR(500),
    metadata JSONB DEFAULT '{}',
    message_count INTEGER DEFAULT 0,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    tool_calls JSONB DEFAULT '[]',
    parent_message_id UUID REFERENCES messages(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agents table
CREATE TABLE IF NOT EXISTS agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    system_prompt TEXT,
    model_config JSONB DEFAULT '{}',
    tools TEXT[] DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'idle',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- MCP Servers table (must come before tools due to foreign key)
CREATE TABLE IF NOT EXISTS mcp_servers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    endpoint VARCHAR(500) NOT NULL,
    authentication JSONB DEFAULT '{}',
    capabilities TEXT[] DEFAULT '{}',
    health_check_url VARCHAR(500),
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Tools table
CREATE TABLE IF NOT EXISTS tools (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    schema JSONB NOT NULL,
    mcp_server_id UUID REFERENCES mcp_servers(id) ON DELETE SET NULL,
    enabled BOOLEAN DEFAULT true,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- External Services table
CREATE TABLE IF NOT EXISTS external_services (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    service_type VARCHAR(100) NOT NULL,
    endpoint VARCHAR(500) NOT NULL,
    authentication JSONB DEFAULT '{}',
    retry_config JSONB DEFAULT '{}',
    circuit_breaker_config JSONB DEFAULT '{}',
    health_check_url VARCHAR(500),
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
CREATE INDEX IF NOT EXISTS idx_agents_name ON agents(name);
CREATE INDEX IF NOT EXISTS idx_tools_enabled ON tools(enabled);
CREATE INDEX IF NOT EXISTS idx_tools_mcp_server_id ON tools(mcp_server_id);
CREATE INDEX IF NOT EXISTS idx_mcp_servers_enabled ON mcp_servers(enabled);
CREATE INDEX IF NOT EXISTS idx_external_services_type ON external_services(service_type);
CREATE INDEX IF NOT EXISTS idx_external_services_enabled ON external_services(enabled);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_sessions_title_gin ON sessions USING gin(title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_messages_content_gin ON messages USING gin(content gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_agents_name_gin ON agents USING gin(name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_agents_description_gin ON agents USING gin(description gin_trgm_ops);

-- Triggers for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_sessions_updated_at BEFORE UPDATE ON sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_messages_updated_at BEFORE UPDATE ON messages
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tools_updated_at BEFORE UPDATE ON tools
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_mcp_servers_updated_at BEFORE UPDATE ON mcp_servers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_external_services_updated_at BEFORE UPDATE ON external_services
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Constraints and validations
ALTER TABLE sessions ADD CONSTRAINT sessions_message_count_positive
    CHECK (message_count >= 0);

ALTER TABLE messages ADD CONSTRAINT messages_role_valid
    CHECK (role IN ('user', 'assistant', 'system', 'tool'));

ALTER TABLE agents ADD CONSTRAINT agents_status_valid
    CHECK (status IN ('idle', 'processing', 'waiting', 'error', 'completed'));

ALTER TABLE external_services ADD CONSTRAINT external_services_type_valid
    CHECK (service_type IN ('llm_provider', 'mcp_server', 'database', 'cache', 'secret_manager', 'message_queue'));

-- Comments for documentation
COMMENT ON TABLE sessions IS 'User conversation sessions';
COMMENT ON TABLE messages IS 'Messages within conversation sessions';
COMMENT ON TABLE agents IS 'AI agent configurations';
COMMENT ON TABLE tools IS 'Tools available to agents';
COMMENT ON TABLE mcp_servers IS 'MCP server configurations';
COMMENT ON TABLE external_services IS 'External service configurations for resilience monitoring';

COMMENT ON COLUMN sessions.metadata IS 'Flexible metadata storage as JSON';
COMMENT ON COLUMN messages.tool_calls IS 'Tool calls made by the message as JSON array';
COMMENT ON COLUMN agents.model_config IS 'LLM model configuration as JSON';
COMMENT ON COLUMN tools.schema IS 'Tool parameter schema as JSON';
COMMENT ON COLUMN mcp_servers.authentication IS 'Authentication configuration as JSON';
COMMENT ON COLUMN external_services.retry_config IS 'Retry configuration as JSON';
COMMENT ON COLUMN external_services.circuit_breaker_config IS 'Circuit breaker configuration as JSON';
