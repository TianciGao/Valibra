"""Pydantic models for inter-service communication.
定义的是“三个微服务之间传递数据的格式”"""

from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class InitTaskRequest(BaseModel): # 用于在数据库环境服务和用户模拟器服务中初始化任务的请求模型
    task_id: str 
    task_data: Dict[str, Any]


class ExecuteSQLRequest(BaseModel): # 用于在数据库环境服务中执行 SQL 查询的请求模型
    sql: str
    task_id: str


class ExecuteSQLResponse(BaseModel): # 用于在数据库环境服务中执行 SQL 查询的响应模型
    result: str
    success: bool
    error: Optional[str] = None


class SubmitSQLRequest(BaseModel): # 用于在数据库环境服务中提交 SQL 查询的请求模型
    sql: str
    task_id: str


class SubmitSQLResponse(BaseModel): # 用于在数据库环境服务中提交 SQL 查询的响应模型
    passed: bool
    message: str
    reward: float = 0.0
    phase_completed: Optional[int] = None
    has_follow_up: bool = False
    follow_up_query: Optional[str] = None


class SchemaRequest(BaseModel): # 用于在数据库环境服务中请求数据库模式信息的请求模型
    task_id: str


class ColumnMeaningRequest(BaseModel): # 用于在数据库环境服务中请求列含义的请求模型
    task_id: str
    table_name: str
    column_name: str


class KnowledgeRequest(BaseModel): # 用于在数据库环境服务中请求知识库信息的请求模型
    task_id: str
    knowledge_name: Optional[str] = None


class AskUserRequest(BaseModel): # 用于在用户模拟器服务中请求用户回答的请求模型
    question: str
    task_id: str


class AskUserResponse(BaseModel): # 用于在用户模拟器服务中返回用户回答的响应模型
    answer: str


class PhaseTransitionRequest(BaseModel): # 用于在用户模拟器服务中请求阶段转换的请求模型 
    task_id: str
