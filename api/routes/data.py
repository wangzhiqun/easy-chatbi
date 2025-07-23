"""
Data routes for ChatBI platform.
Handles data source management, table exploration, and direct data access.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from ..database import get_db, get_table_names, get_table_schema
from ..models import User as UserModel, QueryHistory as QueryHistoryModel
from ..schemas import DataTableSchema, QueryHistory, ChartData, ChartConfig
from .auth import get_current_active_user
from services.data_service import DataService
from ai.tools.sql_executor import SQLExecutor
from utils.logger import get_logger
from utils.exceptions import ChatBIException, DatabaseException

logger = get_logger(__name__)

router = APIRouter()


def get_data_service() -> DataService:
    """Get data service instance."""
    return DataService()


def get_sql_executor() -> SQLExecutor:
    """Get SQL executor instance."""
    return SQLExecutor()


@router.get("/tables", response_model=List[str])
async def get_available_tables(
        current_user: UserModel = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    """Get list of available database tables."""
    try:
        tables = get_table_names()
        logger.info(f"Retrieved {len(tables)} tables for user: {current_user.username}")
        return tables
    except DatabaseException as e:
        logger.error(f"Failed to get table names: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve table information"
        )


@router.get("/tables/{table_name}/schema", response_model=DataTableSchema)
async def get_table_info(
        table_name: str,
        current_user: UserModel = Depends(get_current_active_user),
        db: Session = Depends(get_db),
        data_service: DataService = Depends(get_data_service)
):
    """Get detailed schema information for a specific table."""
    try:
        schema_info = get_table_schema(table_name)

        # Get sample data
        sample_data = await data_service.get_sample_data(table_name, limit=5)

        return DataTableSchema(
            table_name=table_name,
            columns=schema_info["columns"],
            sample_data=sample_data,
            business_context=f"Table containing {table_name} data"
        )

    except DatabaseException as e:
        logger.error(f"Failed to get table schema for {table_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Table '{table_name}' not found or inaccessible"
        )


@router.get("/tables/{table_name}/sample")
async def get_table_sample(
        table_name: str,
        current_user: UserModel = Depends(get_current_active_user),
        limit: int = Query(10, ge=1, le=100),
        data_service: DataService = Depends(get_data_service)
):
    """Get sample data from a table."""
    try:
        sample_data = await data_service.get_sample_data(table_name, limit=limit)
        return {
            "table_name": table_name,
            "sample_data": sample_data,
            "row_count": len(sample_data)
        }

    except Exception as e:
        logger.error(f"Failed to get sample data for {table_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve sample data"
        )


@router.post("/execute-sql")
async def execute_sql_query(
        sql_query: str,
        current_user: UserModel = Depends(get_current_active_user),
        db: Session = Depends(get_db),
        sql_executor: SQLExecutor = Depends(get_sql_executor)
):
    """Execute a SQL query with safety checks."""
    try:
        # Execute SQL with safety validation
        result = await sql_executor.execute_query(
            sql_query=sql_query,
            user_id=current_user.id
        )

        # Log query execution
        query_log = QueryHistoryModel(
            user_id=current_user.id,
            user_question="Direct SQL execution",
            generated_sql=sql_query,
            execution_status=result["status"],
            execution_time_ms=result.get("execution_time_ms"),
            result_rows=len(result.get("data", [])),
            is_safe=result.get("is_safe", True)
        )
        db.add(query_log)
        db.commit()

        logger.info(f"SQL executed by user {current_user.username}: {sql_query[:100]}...")
        return result

    except ChatBIException as e:
        logger.error(f"SQL execution failed: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )


@router.get("/query-history", response_model=List[QueryHistory])
async def get_query_history(
        current_user: UserModel = Depends(get_current_active_user),
        db: Session = Depends(get_db),
        skip: int = Query(0, ge=0),
        limit: int = Query(20, ge=1, le=100),
        status_filter: Optional[str] = Query(None, regex="^(success|error|pending)$")
):
    """Get user's query history."""
    query = db.query(QueryHistoryModel).filter(
        QueryHistoryModel.user_id == current_user.id
    )

    if status_filter:
        query = query.filter(QueryHistoryModel.execution_status == status_filter)

    queries = query.order_by(QueryHistoryModel.created_at.desc()).offset(skip).limit(limit).all()

    return [QueryHistory.from_orm(q) for q in queries]


@router.get("/statistics")
async def get_data_statistics(
        current_user: UserModel = Depends(get_current_active_user),
        db: Session = Depends(get_db),
        data_service: DataService = Depends(get_data_service)
):
    """Get data statistics and insights."""
    try:
        # Get table count
        tables = get_table_names()
        table_count = len(tables)

        # Get user query statistics
        total_queries = db.query(QueryHistoryModel).filter(
            QueryHistoryModel.user_id == current_user.id
        ).count()

        successful_queries = db.query(QueryHistoryModel).filter(
            QueryHistoryModel.user_id == current_user.id,
            QueryHistoryModel.execution_status == "success"
        ).count()

        # Get recent activity
        recent_queries = db.query(QueryHistoryModel).filter(
            QueryHistoryModel.user_id == current_user.id
        ).order_by(QueryHistoryModel.created_at.desc()).limit(5).all()

        return {
            "table_count": table_count,
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "success_rate": (successful_queries / total_queries * 100) if total_queries > 0 else 0,
            "recent_queries": [
                {
                    "question": q.user_question,
                    "status": q.execution_status,
                    "created_at": q.created_at
                } for q in recent_queries
            ]
        }

    except Exception as e:
        logger.error(f"Failed to get data statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )


@router.post("/visualize", response_model=ChartData)
async def create_visualization(
        chart_config: ChartConfig,
        sql_query: str,
        current_user: UserModel = Depends(get_current_active_user),
        sql_executor: SQLExecutor = Depends(get_sql_executor)
):
    """Create a visualization from SQL query results."""
    try:
        # Execute SQL query
        result = await sql_executor.execute_query(
            sql_query=sql_query,
            user_id=current_user.id
        )

        if result["status"] != "success":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Query execution failed: {result.get('error', 'Unknown error')}"
            )

        # Validate chart configuration against data
        data = result["data"]
        if not data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No data returned from query"
            )

        # Check if required columns exist
        columns = list(data[0].keys()) if data else []
        if chart_config.x_axis not in columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"X-axis column '{chart_config.x_axis}' not found in data"
            )

        y_columns = [chart_config.y_axis] if isinstance(chart_config.y_axis, str) else chart_config.y_axis
        for y_col in y_columns:
            if y_col not in columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Y-axis column '{y_col}' not found in data"
                )

        return ChartData(
            config=chart_config,
            data=data,
            metadata={
                "row_count": len(data),
                "columns": columns,
                "sql_query": sql_query
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visualization creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create visualization"
        )


@router.get("/export/{table_name}")
async def export_table_data(
        table_name: str,
        current_user: UserModel = Depends(get_current_active_user),
        format: str = Query("csv", regex="^(csv|json|excel)$"),
        limit: Optional[int] = Query(None, ge=1, le=10000),
        data_service: DataService = Depends(get_data_service)
):
    """Export table data in various formats."""
    try:
        # Get table data
        data = await data_service.get_table_data(
            table_name=table_name,
            limit=limit or 1000
        )

        if format == "csv":
            return data_service.export_to_csv(data)
        elif format == "json":
            return data_service.export_to_json(data)
        elif format == "excel":
            return data_service.export_to_excel(data, table_name)

    except Exception as e:
        logger.error(f"Data export failed for {table_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export data"
        )