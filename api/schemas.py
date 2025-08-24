from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
from pydantic import BaseModel, Field, EmailStr, field_serializer


class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr


class UserCreate(UserBase):
    password: str = Field(..., min_length=6)


class UserUpdate(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = None


class UserResponse(UserBase):
    id: int
    is_active: bool
    api_key: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationCreate(BaseModel):
    title: Optional[str] = Field(None, max_length=200)


class ConversationResponse(BaseModel):
    id: str
    user_id: Optional[int]
    title: Optional[str]
    context: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MessageCreate(BaseModel):
    content: str = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = None


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    metadata: Optional[Dict[str, Any]]
    created_at: datetime

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    metadata: Optional[Dict[str, Any]] = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    database: Optional[str] = None


class QueryResponse(BaseModel):
    status: str
    query: str
    row_count: Optional[int]
    columns: Optional[List[str]]
    data: Optional[List[Dict[str, Any]]]
    execution_time: Optional[int] = Field(default=datetime.now())
    error: Optional[str] = None


class TableInfoRequest(BaseModel):
    table_name: str = Field(..., min_length=1)


class TableInfoResponse(BaseModel):
    table_name: str
    columns: List[Dict[str, Any]]
    row_count: int
    comment: Optional[str]
    sample_data: Optional[List[Dict[str, Any]]]


class SchemaResponse(BaseModel):
    database: str
    tables: Dict[str, Any]


class ChartRequest(BaseModel):
    data: List[Dict[str, Any]]
    chart_type: str = Field(default="auto")
    options: Optional[Dict[str, Any]] = None


class ChartResponse(BaseModel):
    status: str
    chart_type: str
    config: Dict[str, Any]
    data_points: int


class AnalysisRequest(BaseModel):
    data: List[Dict[str, Any]]
    analysis_type: str = Field(default="comprehensive")


class AnalysisResponse(BaseModel):
    status: str
    analysis_type: str
    results: Dict[str, Any]
    data_points: int

    @field_serializer("results")
    def serialize_results(self, value):
        def convert(v):
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            return v

        return {k: convert(v) for k, v in value.items()}


class MCPToolRequest(BaseModel):
    tool_name: str
    arguments: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


class MCPResourceRequest(BaseModel):
    resource_uri: str
    session_id: Optional[str] = None


class MCPPromptRequest(BaseModel):
    prompt_name: str
    arguments: Dict[str, str]
    session_id: Optional[str] = None


class MCPToolResponse(BaseModel):
    status: str
    result: Any


class MCPResourceResponse(BaseModel):
    status: str
    content: Any


class MCPPromptResponse(BaseModel):
    status: str
    prompt: str


class ExportRequest(BaseModel):
    query: str
    format: str = Field(default="csv", pattern="^(csv|json|excel)$")


class ExportResponse(BaseModel):
    status: str
    format: str
    content_type: str
    data: Any
    row_count: int


class KnowledgeCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    category: Optional[str] = None
    tags: Optional[List[str]] = None


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class KnowledgeResponse(BaseModel):
    id: int
    title: str
    content: str
    category: Optional[str]
    tags: Optional[List[str]]
    created_at: datetime

    class Config:
        from_attributes = True


class ErrorResponse(BaseModel):
    error: str
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class GenerateSQLRequest(BaseModel):
    description: str
    include_explanation: bool = True
    query_optimization: bool = False
    result_limit: int = 100
    safety_check: bool = True


class GenerateSQLResponse(BaseModel):
    status: str
    sql: Optional[str] = None
    explanation: Optional[str] = None
    error: Optional[str] = None


class ValidateRequest(BaseModel):
    query: str
