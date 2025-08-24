import io

import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ai import LLMClient
from ai.agents import ChartAgent, AnalysisAgent, SQLAgent
from core import DataService
from utils import logger
from ..database import get_db
from ..schemas import (
    QueryRequest, QueryResponse,
    TableInfoRequest, TableInfoResponse,
    SchemaResponse, ExportRequest, ChartRequest, ChartResponse,
    AnalysisRequest, AnalysisResponse,
    GenerateSQLRequest, GenerateSQLResponse
)

router = APIRouter()
data_service = DataService()
chart_agent = ChartAgent()
analysis_agent = AnalysisAgent()
llm_client = LLMClient()
sql_agent = SQLAgent(llm_client)


@router.post("/query", response_model=QueryResponse)
async def execute_query(request: QueryRequest, db: Session = Depends(get_db)):
    try:
        result = await data_service.execute_query(
            request.query,
            {'database': request.database} if request.database else None
        )

        return QueryResponse(**result)

    except Exception as e:
        logger.error(f"Query execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schema", response_model=SchemaResponse)
async def get_schema(db: Session = Depends(get_db)):
    try:
        schema = await data_service.get_schema()
        return SchemaResponse(**schema)

    except Exception as e:
        logger.error(f"Failed to get schema: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/table-info", response_model=TableInfoResponse)
async def get_table_info(request: TableInfoRequest, db: Session = Depends(get_db)):
    try:
        info = await data_service.get_table_info(request.table_name)
        return TableInfoResponse(**info)

    except Exception as e:
        logger.error(f"Failed to get table info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export")
async def export_data(request: ExportRequest, db: Session = Depends(get_db)):
    try:
        result = await data_service.export_data(request.query, request.format)

        if result['status'] != 'success':
            raise HTTPException(status_code=400, detail=result.get('error'))

        if request.format == 'csv':
            return StreamingResponse(
                io.StringIO(result['data']),
                media_type=result['content_type'],
                headers={"Content-Disposition": "attachment; filename=export.csv"}
            )
        elif request.format == 'json':
            return StreamingResponse(
                io.StringIO(result['data']),
                media_type=result['content_type'],
                headers={"Content-Disposition": "attachment; filename=export.json"}
            )
        elif request.format == 'excel':
            return StreamingResponse(
                io.BytesIO(result['data']),
                media_type=result['content_type'],
                headers={"Content-Disposition": "attachment; filename=export.xlsx"}
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import")
async def import_data(
        table_name: str,
        file: UploadFile = File(...),
        db: Session = Depends(get_db)
):
    try:
        content = await file.read()

        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(content))
        elif file.filename.endswith('.json'):
            df = pd.read_json(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        result = await data_service.import_data(table_name, df)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Import failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chart", response_model=ChartResponse)
async def create_chart(request: ChartRequest, db: Session = Depends(get_db)):
    try:
        df = pd.DataFrame(request.data)

        if request.chart_type == 'auto':
            recommendation = chart_agent.recommend_chart(df)
            chart_type = recommendation['chart_type']
        else:
            chart_type = request.chart_type

        config = chart_agent.generate_chart_config(
            df,
            chart_type,
            **(request.options or {})
        )

        return ChartResponse(
            status='success',
            chart_type=chart_type,
            config=config,
            data_points=len(request.data)
        )

    except Exception as e:
        logger.error(f"Chart creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_data(request: AnalysisRequest, db: Session = Depends(get_db)):
    try:
        df = pd.DataFrame(request.data)

        if request.analysis_type == 'comprehensive':
            results = analysis_agent.analyze_data(df)
        elif request.analysis_type == 'correlation':
            results = analysis_agent.find_correlations(df)
        elif request.analysis_type == 'anomaly':
            results = analysis_agent.detect_anomalies(df)
        elif request.analysis_type == 'trend':
            results = analysis_agent.trend_analysis(df)
        else:
            results = analysis_agent.analyze_data(df, request.analysis_type)

        return AnalysisResponse(
            status='success',
            analysis_type=request.analysis_type,
            results=data_service.clean_numpy_types(results),
            data_points=int(len(request.data))
        )

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_query_history(limit: int = 10, db: Session = Depends(get_db)):
    try:
        history = await data_service.get_query_history(limit)
        return history

    except Exception as e:
        logger.error(f"Failed to get query history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate-sql")
async def validate_sql(request: QueryRequest, db: Session = Depends(get_db)):
    try:
        from core import SecurityManager
        security = SecurityManager()

        is_valid, error = security.validate_sql_query(request.query)

        return {
            'valid': is_valid,
            'error': error,
            'query': request.query
        }

    except Exception as e:
        logger.error(f"SQL validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-sql")
async def generate_sql_from_description(request: GenerateSQLRequest):
    try:
        logger.info(f"Generating SQL from description: {request.description[:100]}...")

        schema = await data_service.get_schema()

        sql_query, explanation = sql_agent.generate_sql(
            question=request.description,
            schema_info=schema
        )

        logger.info(f"Generated SQL: {sql_query}")

        if not sql_query:
            return GenerateSQLResponse(
                status="error",
                error="无法根据描述生成SQL语句，请提供更详细的信息"
            )

        final_explanation = explanation if request.include_explanation else None

        return GenerateSQLResponse(
            status="success",
            sql=sql_query,
            explanation=final_explanation
        )

    except Exception as e:
        logger.error(f"Error generating SQL: {str(e)}")
        return GenerateSQLResponse(
            status="error",
            error=f"生成SQL时发生错误: {str(e)}"
        )
