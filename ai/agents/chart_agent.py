import json
from typing import Dict, Any, Optional, List

import pandas as pd

from utils import logger, AIError
from ..llm_client import LLMClient
from ..prompts import PromptTemplates
from ..tools import AnalysisTool


class ChartAgent:
    CHART_TYPES = {
        'line': '折线图',
        'bar': '柱状图',
        'scatter': '散点图',
        'pie': '饼图',
        'heatmap': '热力图',
        'area': '面积图',
        'box': '箱线图',
        'histogram': '直方图'
    }

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self.analysis_tool = AnalysisTool()
        logger.info("Initialized Chart Agent")

    def recommend_chart(
            self,
            df: pd.DataFrame,
            user_request: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            data_info = self._prepare_data_info(df)

            system_prompt, user_prompt = PromptTemplates.get_chart_prompt(
                data_info=data_info,
                user_request=user_request or "自动选择最佳图表类型"
            )

            response = self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.5
            )

            recommendation = self._parse_recommendation(response, df)

            logger.info(f"Recommended chart type: {recommendation['chart_type']}")
            return recommendation

        except Exception as e:
            logger.error(f"Chart recommendation failed: {str(e)}")
            return self._fallback_recommendation(df)

    def generate_chart_config(
            self,
            df: pd.DataFrame,
            chart_type: str,
            **kwargs
    ) -> Dict[str, Any]:
        try:
            if chart_type not in self.CHART_TYPES:
                raise ValueError(f"Unsupported chart type: {chart_type}")

            if chart_type == 'line':
                config = self._generate_line_config(df, **kwargs)
            elif chart_type == 'bar':
                config = self._generate_bar_config(df, **kwargs)
            elif chart_type == 'scatter':
                config = self._generate_scatter_config(df, **kwargs)
            elif chart_type == 'pie':
                config = self._generate_pie_config(df, **kwargs)
            elif chart_type == 'heatmap':
                config = self._generate_heatmap_config(df, **kwargs)
            elif chart_type == 'area':
                config = self._generate_area_config(df, **kwargs)
            elif chart_type == 'box':
                config = self._generate_box_config(df, **kwargs)
            elif chart_type == 'histogram':
                config = self._generate_histogram_config(df, **kwargs)
            else:
                config = self._generate_default_config(df, chart_type, **kwargs)

            logger.info(f"Generated {chart_type} chart configuration")
            return config

        except Exception as e:
            logger.error(f"Chart config generation failed: {str(e)}")
            raise AIError(f"Chart config generation failed: {str(e)}")

    def enhance_chart(self, config: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = f"""
优化以下图表配置，使其更美观和交互性更强：

当前配置：
{json.dumps(config, ensure_ascii=False, indent=2)}

请提供优化建议，包括：
1. 颜色方案
2. 标题和标签
3. 交互功能
4. 布局优化

输出JSON格式的优化配置：
"""

            response = self.llm.generate(
                prompt=prompt,
                temperature=0.5
            )

            enhanced = self._parse_json_response(response)

            config.update(enhanced)

            logger.info("Chart configuration enhanced")
            return config

        except Exception as e:
            logger.error(f"Chart enhancement failed: {str(e)}")
            return config

    def _prepare_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            'row_count': len(df),
            'col_count': len(df.columns),
            'columns': list(df.columns),
            'data_types': str(df.dtypes.to_dict()),
            'data_sample': df.head(5).to_string() if len(df) > 0 else "Empty DataFrame",
            'statistics': self.analysis_tool.generate_statistics(df)
        }

    def _parse_recommendation(self, response: str, df: pd.DataFrame) -> Dict[str, Any]:
        recommendation = {
            'chart_type': 'bar',
            'reason': '',
            'config': {},
            'suggestions': []
        }

        for chart_type, chinese_name in self.CHART_TYPES.items():
            if chinese_name in response or chart_type in response.lower():
                recommendation['chart_type'] = chart_type
                break

        lines = response.split('\n')
        for i, line in enumerate(lines):
            if 'X轴' in line or 'x轴' in line:
                recommendation['config']['x_axis'] = self._extract_column_name(line, df.columns)
            elif 'Y轴' in line or 'y轴' in line:
                recommendation['config']['y_axis'] = self._extract_column_name(line, df.columns)
            elif '理由' in line or '原因' in line:
                if i + 1 < len(lines):
                    recommendation['reason'] = lines[i + 1].strip()

        if '建议' in response:
            suggestion_start = response.index('建议')
            suggestion_text = response[suggestion_start:]
            recommendation['suggestions'] = [
                                                s.strip() for s in suggestion_text.split('\n')
                                                if s.strip() and not s.strip().startswith('建议')
                                            ][:3]

        return recommendation

    def _extract_column_name(self, text: str, columns: List[str]) -> Optional[str]:
        for col in columns:
            if col in text:
                return col
        return None

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except:
            pass

        return {}

    def _fallback_recommendation(self, df: pd.DataFrame) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        recommendation = {
            'chart_type': 'bar',
            'reason': 'Default recommendation based on data structure',
            'config': {},
            'suggestions': []
        }

        if len(numeric_cols) >= 2:
            if len(df) > 50:
                recommendation['chart_type'] = 'scatter'
                recommendation['config'] = {
                    'x_axis': numeric_cols[0],
                    'y_axis': numeric_cols[1]
                }
            else:
                recommendation['chart_type'] = 'line'
                recommendation['config'] = {
                    'x_axis': df.index.name or 'index',
                    'y_axis': numeric_cols[0]
                }
        elif len(cat_cols) >= 1 and len(numeric_cols) >= 1:
            recommendation['chart_type'] = 'bar'
            recommendation['config'] = {
                'x_axis': cat_cols[0],
                'y_axis': numeric_cols[0]
            }

        return recommendation

    def _generate_line_config(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        config = {
            'type': 'line',
            'data': df.to_dict('records'),
            'x': kwargs.get('x_axis', df.index.name or 'index'),
            'y': kwargs.get('y_axis', numeric_cols[0] if numeric_cols else None),
            'title': kwargs.get('title', 'Line Chart'),
            'xaxis_title': kwargs.get('xaxis_title', ''),
            'yaxis_title': kwargs.get('yaxis_title', ''),
            'show_legend': True,
            'markers': True
        }

        return config

    def _generate_bar_config(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        config = {
            'type': 'bar',
            'data': df.to_dict('records'),
            'x': kwargs.get('x_axis', cat_cols[0] if cat_cols else df.index.name),
            'y': kwargs.get('y_axis', numeric_cols[0] if numeric_cols else None),
            'title': kwargs.get('title', 'Bar Chart'),
            'orientation': kwargs.get('orientation', 'v'),
            'color': kwargs.get('color', None)
        }

        return config

    def _generate_scatter_config(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        config = {
            'type': 'scatter',
            'data': df.to_dict('records'),
            'x': kwargs.get('x_axis', numeric_cols[0] if len(numeric_cols) > 0 else None),
            'y': kwargs.get('y_axis', numeric_cols[1] if len(numeric_cols) > 1 else None),
            'title': kwargs.get('title', 'Scatter Plot'),
            'size': kwargs.get('size', None),
            'color': kwargs.get('color', None),
            'hover_data': kwargs.get('hover_data', [])
        }

        return config

    def _generate_pie_config(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        config = {
            'type': 'pie',
            'data': df.to_dict('records'),
            'values': kwargs.get('values', numeric_cols[0] if numeric_cols else None),
            'names': kwargs.get('names', cat_cols[0] if cat_cols else df.index.name),
            'title': kwargs.get('title', 'Pie Chart'),
            'hole': kwargs.get('hole', 0)
        }

        return config

    def _generate_heatmap_config(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:

        numeric_df = df.select_dtypes(include=['number'])

        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            data = corr_matrix.values.tolist()
            x_labels = corr_matrix.columns.tolist()
            y_labels = corr_matrix.index.tolist()
        else:
            data = df.values.tolist()
            x_labels = df.columns.tolist()
            y_labels = df.index.tolist()

        config = {
            'type': 'heatmap',
            'z': data,
            'x': x_labels,
            'y': y_labels,
            'title': kwargs.get('title', 'Heatmap'),
            'colorscale': kwargs.get('colorscale', 'Viridis')
        }

        return config

    def _generate_area_config(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        config = self._generate_line_config(df, **kwargs)
        config['type'] = 'area'
        config['fill'] = 'tozeroy'
        return config

    def _generate_box_config(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        config = {
            'type': 'box',
            'data': df.to_dict('records'),
            'y': kwargs.get('y_axis', numeric_cols[0] if numeric_cols else None),
            'title': kwargs.get('title', 'Box Plot'),
            'boxpoints': kwargs.get('boxpoints', 'outliers')
        }

        return config

    def _generate_histogram_config(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        config = {
            'type': 'histogram',
            'data': df.to_dict('records'),
            'x': kwargs.get('x_axis', numeric_cols[0] if numeric_cols else None),
            'title': kwargs.get('title', 'Histogram'),
            'nbins': kwargs.get('nbins', 20)
        }

        return config

    def _generate_default_config(
            self,
            df: pd.DataFrame,
            chart_type: str,
            **kwargs
    ) -> Dict[str, Any]:
        return {
            'type': chart_type,
            'data': df.to_dict('records'),
            'title': kwargs.get('title', f'{chart_type.capitalize()} Chart'),
            **kwargs
        }

    def generate_chart(self, df: pd.DataFrame, chart_type: str, user_request: str = "") -> Dict[str, Any]:
        config = {
            'type': chart_type,
            'title': self._generate_chart_title(df, chart_type, user_request),
            'theme': 'light'
        }

        if chart_type == 'bar':
            return self._generate_bar_chart(df, config)
        elif chart_type == 'line':
            return self._generate_line_chart(df, config)
        elif chart_type == 'pie':
            return self._generate_pie_chart(df, config)
        elif chart_type == 'table':
            return self._generate_table_chart(df, config)
        else:
            return self._generate_table_chart(df, config)

    def _generate_bar_chart(self, df: pd.DataFrame, config: dict) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()

        if not numeric_cols or not categorical_cols:
            return self._generate_table_chart(df, config)

        x_col = categorical_cols[0]
        y_col = numeric_cols[0]

        chart_data = df.groupby(x_col)[y_col].sum().reset_index()

        config.update({
            'x_axis': x_col,
            'y_axis': y_col,
            'x_title': x_col,
            'y_title': y_col
        })

        data = {
            'labels': chart_data[x_col].tolist(),
            'datasets': [{
                'label': y_col,
                'data': chart_data[y_col].tolist(),
                'backgroundColor': self._generate_colors(len(chart_data))
            }]
        }

        return {'config': config, 'data': data}

    def _generate_line_chart(self, df: pd.DataFrame, config: dict) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) < 2:
            return self._generate_table_chart(df, config)

        x_col = numeric_cols[0]
        y_col = numeric_cols[1]

        chart_data = df.sort_values(x_col)

        config.update({
            'x_axis': x_col,
            'y_axis': y_col,
            'x_title': x_col,
            'y_title': y_col
        })

        data = {
            'labels': chart_data[x_col].tolist(),
            'datasets': [{
                'label': y_col,
                'data': chart_data[y_col].tolist(),
                'borderColor': '#3498db',
                'backgroundColor': 'rgba(52, 152, 219, 0.1)',
                'fill': True
            }]
        }

        return {'config': config, 'data': data}

    def _generate_pie_chart(self, df: pd.DataFrame, config: dict) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()

        if not numeric_cols or not categorical_cols:
            return self._generate_table_chart(df, config)

        category_col = categorical_cols[0]
        value_col = numeric_cols[0]

        chart_data = df.groupby(category_col)[value_col].sum().reset_index()

        config.update({
            'category_field': category_col,
            'value_field': value_col
        })

        data = {
            'labels': chart_data[category_col].tolist(),
            'datasets': [{
                'data': chart_data[value_col].tolist(),
                'backgroundColor': self._generate_colors(len(chart_data))
            }]
        }

        return {'config': config, 'data': data}

    def _generate_table_chart(self, df: pd.DataFrame, config: dict) -> Dict[str, Any]:
        config.update({
            'type': 'table',
            'columns': df.columns.tolist()
        })

        data = {
            'columns': df.columns.tolist(),
            'rows': df.head(100).to_dict('records')
        }

        return {'config': config, 'data': data}

    def _generate_colors(self, count: int) -> List[str]:
        colors = [
            '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
            '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#8e44ad'
        ]
        return (colors * ((count // len(colors)) + 1))[:count]

    def _generate_chart_title(self, df: pd.DataFrame, chart_type: str, user_request: str) -> str:
        if user_request:
            prompt = f"为以下图表生成一个简洁的中文标题：图表类型={chart_type}, 用户请求={user_request}, 数据列={list(df.columns)}"
            try:
                title = self.llm.generate(prompt, max_tokens=50)
                return title.strip()
            except:
                pass

        return f"数据{chart_type.title()}图"
