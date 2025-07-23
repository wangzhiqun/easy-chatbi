"""
Chart Agent for ChatBI platform.
Handles intelligent chart type recommendations and visualization configurations.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
import re

from .llm_client import LLMClient
from .prompts import PromptTemplates
from utils.logger import get_logger
from utils.exceptions import DataProcessingException, ErrorCodes

logger = get_logger(__name__)


class ChartAgent:
    """
    Intelligent chart recommendation agent that analyzes data characteristics
    and user intent to suggest optimal visualizations.
    """

    def __init__(self):
        """Initialize chart agent."""
        self.llm_client = LLMClient()

        # Chart type compatibility matrix
        self.chart_rules = {
            "bar": {
                "best_for": ["categorical_comparison", "ranking", "distribution"],
                "data_types": ["categorical", "numerical"],
                "max_categories": 20,
                "min_data_points": 1
            },
            "line": {
                "best_for": ["time_series", "trends", "continuous_data"],
                "data_types": ["temporal", "numerical"],
                "max_categories": float('inf'),
                "min_data_points": 2
            },
            "pie": {
                "best_for": ["parts_of_whole", "composition"],
                "data_types": ["categorical", "numerical"],
                "max_categories": 8,
                "min_data_points": 2
            },
            "scatter": {
                "best_for": ["correlation", "relationship", "distribution"],
                "data_types": ["numerical", "numerical"],
                "max_categories": float('inf'),
                "min_data_points": 5
            },
            "area": {
                "best_for": ["time_series", "cumulative", "volume"],
                "data_types": ["temporal", "numerical"],
                "max_categories": 10,
                "min_data_points": 3
            },
            "histogram": {
                "best_for": ["distribution", "frequency"],
                "data_types": ["numerical"],
                "max_categories": float('inf'),
                "min_data_points": 10
            }
        }

    async def recommend_chart(
            self,
            data: List[Dict[str, Any]],
            user_question: str,
            sql_query: str,
            column_metadata: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Recommend the best chart type and configuration for the given data.

        Args:
            data: Query results data
            user_question: Original user question
            sql_query: SQL query that generated the data
            column_metadata: Metadata about columns (types, etc.)

        Returns:
            Chart recommendation with configuration
        """
        logger.info(f"Generating chart recommendation for {len(data)} data points")

        try:
            if not data:
                return self._create_empty_chart_response()

            # Analyze data characteristics
            data_analysis = self._analyze_data_characteristics(data, column_metadata)

            # Determine user intent from question
            user_intent = self._analyze_user_intent(user_question)

            # Get rule-based recommendation
            rule_based_rec = self._apply_chart_rules(data_analysis, user_intent)

            # Get AI-based recommendation
            ai_recommendation = await self._get_ai_recommendation(
                user_question=user_question,
                sql_query=sql_query,
                sample_data=data[:5],
                column_info=data_analysis["columns"]
            )

            # Combine recommendations
            final_recommendation = self._combine_recommendations(
                rule_based_rec, ai_recommendation, data_analysis
            )

            # Generate chart configuration
            chart_config = self._create_chart_configuration(
                recommendation=final_recommendation,
                data_analysis=data_analysis,
                user_question=user_question
            )

            logger.info(f"Recommended chart type: {chart_config['chart_type']}")
            return chart_config

        except Exception as e:
            logger.error(f"Chart recommendation failed: {e}")
            return self._create_fallback_chart_response(data)

    def _analyze_data_characteristics(
            self,
            data: List[Dict[str, Any]],
            column_metadata: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Analyze characteristics of the data to inform chart selection."""
        if not data:
            return {"columns": [], "row_count": 0, "data_types": {}}

        df = pd.DataFrame(data)
        analysis = {
            "row_count": len(data),
            "column_count": len(df.columns),
            "columns": [],
            "data_types": {},
            "patterns": {}
        }

        for column in df.columns:
            col_analysis = self._analyze_column(df[column], column)
            analysis["columns"].append(col_analysis)
            analysis["data_types"][column] = col_analysis["data_type"]

        # Detect patterns
        analysis["patterns"] = self._detect_data_patterns(df, analysis["columns"])

        return analysis

    def _analyze_column(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Analyze individual column characteristics."""
        analysis = {
            "name": column_name,
            "data_type": "unknown",
            "is_numeric": False,
            "is_temporal": False,
            "is_categorical": False,
            "unique_count": series.nunique(),
            "null_count": series.isnull().sum(),
            "sample_values": series.dropna().head(5).tolist()
        }

        # Determine data type
        if pd.api.types.is_numeric_dtype(series):
            analysis["data_type"] = "numerical"
            analysis["is_numeric"] = True
            analysis["min_value"] = series.min()
            analysis["max_value"] = series.max()
            analysis["mean_value"] = series.mean()

        elif pd.api.types.is_datetime64_any_dtype(series):
            analysis["data_type"] = "temporal"
            analysis["is_temporal"] = True
            analysis["date_range"] = (series.min(), series.max())

        else:
            # Check if it's a date string
            if self._is_date_column(series):
                analysis["data_type"] = "temporal"
                analysis["is_temporal"] = True
            else:
                analysis["data_type"] = "categorical"
                analysis["is_categorical"] = True

        # Determine if column is suitable for grouping
        total_rows = len(series)
        if analysis["unique_count"] <= total_rows * 0.5 and analysis["unique_count"] <= 50:
            analysis["good_for_grouping"] = True
        else:
            analysis["good_for_grouping"] = False

        return analysis

    def _is_date_column(self, series: pd.Series) -> bool:
        """Check if a text column contains date values."""
        sample = series.dropna().head(10)
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
            r'\w+ \d{1,2}, \d{4}'  # Month DD, YYYY
        ]

        for value in sample:
            if isinstance(value, str):
                for pattern in date_patterns:
                    if re.search(pattern, value):
                        return True
        return False

    def _detect_data_patterns(self, df: pd.DataFrame, columns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect common data patterns that influence chart selection."""
        patterns = {
            "has_time_series": False,
            "has_categories": False,
            "has_numerical": False,
            "is_aggregated": False,
            "relationship_data": False
        }

        # Check for time series
        temporal_cols = [col for col in columns if col["is_temporal"]]
        if temporal_cols:
            patterns["has_time_series"] = True

        # Check for categorical data
        categorical_cols = [col for col in columns if col["is_categorical"]]
        if categorical_cols:
            patterns["has_categories"] = True

        # Check for numerical data
        numerical_cols = [col for col in columns if col["is_numeric"]]
        if numerical_cols:
            patterns["has_numerical"] = True

        # Check if data looks aggregated (common aggregation column names)
        agg_indicators = ["count", "sum", "total", "avg", "average", "min", "max"]
        for col in columns:
            if any(indicator in col["name"].lower() for indicator in agg_indicators):
                patterns["is_aggregated"] = True
                break

        # Check for relationship data (multiple numerical columns)
        if len(numerical_cols) >= 2:
            patterns["relationship_data"] = True

        return patterns

    def _analyze_user_intent(self, user_question: str) -> Dict[str, Any]:
        """Analyze user question to understand visualization intent."""
        question_lower = user_question.lower()

        intent = {
            "comparison": False,
            "trend": False,
            "distribution": False,
            "relationship": False,
            "composition": False,
            "ranking": False
        }

        # Comparison indicators
        comparison_words = ["compare", "versus", "vs", "difference", "between"]
        if any(word in question_lower for word in comparison_words):
            intent["comparison"] = True

        # Trend indicators
        trend_words = ["trend", "over time", "change", "growth", "decline", "increase", "decrease"]
        if any(word in question_lower for word in trend_words):
            intent["trend"] = True

        # Distribution indicators
        distribution_words = ["distribution", "spread", "range", "frequency", "histogram"]
        if any(word in question_lower for word in distribution_words):
            intent["distribution"] = True

        # Relationship indicators
        relationship_words = ["relationship", "correlation", "impact", "affect", "influence"]
        if any(word in question_lower for word in relationship_words):
            intent["relationship"] = True

        # Composition indicators
        composition_words = ["breakdown", "composition", "percentage", "proportion", "share"]
        if any(word in question_lower for word in composition_words):
            intent["composition"] = True

        # Ranking indicators
        ranking_words = ["top", "bottom", "best", "worst", "highest", "lowest", "rank"]
        if any(word in question_lower for word in ranking_words):
            intent["ranking"] = True

        return intent

    def _apply_chart_rules(
            self,
            data_analysis: Dict[str, Any],
            user_intent: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        """Apply rule-based logic to recommend chart types with confidence scores."""
        recommendations = []
        patterns = data_analysis["patterns"]
        row_count = data_analysis["row_count"]
        columns = data_analysis["columns"]

        for chart_type, rules in self.chart_rules.items():
            confidence = 0.0

            # Check data point requirements
            if row_count < rules["min_data_points"]:
                continue

            # Check category limits
            categorical_cols = [col for col in columns if col["is_categorical"]]
            if categorical_cols:
                max_categories = max(col["unique_count"] for col in categorical_cols)
                if max_categories > rules["max_categories"]:
                    continue

            # Score based on data patterns
            if chart_type == "line" and patterns["has_time_series"]:
                confidence += 0.8

            if chart_type == "bar" and patterns["has_categories"] and patterns["has_numerical"]:
                confidence += 0.7

            if chart_type == "pie" and patterns["has_categories"] and len(categorical_cols) == 1:
                confidence += 0.6

            if chart_type == "scatter" and patterns["relationship_data"]:
                confidence += 0.8

            if chart_type == "area" and patterns["has_time_series"]:
                confidence += 0.6

            if chart_type == "histogram" and patterns["has_numerical"] and not patterns["has_categories"]:
                confidence += 0.7

            # Score based on user intent
            if user_intent["trend"] and chart_type in ["line", "area"]:
                confidence += 0.3

            if user_intent["comparison"] and chart_type == "bar":
                confidence += 0.3

            if user_intent["composition"] and chart_type == "pie":
                confidence += 0.3

            if user_intent["relationship"] and chart_type == "scatter":
                confidence += 0.3

            if user_intent["distribution"] and chart_type in ["histogram", "bar"]:
                confidence += 0.3

            if confidence > 0:
                recommendations.append((chart_type, confidence))

        return sorted(recommendations, key=lambda x: x[1], reverse=True)

    async def _get_ai_recommendation(
            self,
            user_question: str,
            sql_query: str,
            sample_data: List[Dict[str, Any]],
            column_info: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Get AI-based chart recommendation."""
        try:
            recommendation = await self.llm_client.suggest_chart_type(
                sql_query=sql_query,
                sample_data=sample_data,
                user_question=user_question
            )
            return recommendation
        except Exception as e:
            logger.warning(f"AI chart recommendation failed: {e}")
            return {"chart_type": "bar", "explanation": "Default recommendation"}

    def _combine_recommendations(
            self,
            rule_based: List[Tuple[str, float]],
            ai_recommendation: Dict[str, Any],
            data_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine rule-based and AI recommendations."""
        # Start with rule-based recommendation
        if rule_based:
            primary_rec = rule_based[0]
            chart_type = primary_rec[0]
            confidence = primary_rec[1]
        else:
            chart_type = "bar"  # Default
            confidence = 0.5

        # Consider AI recommendation
        ai_chart_type = ai_recommendation.get("chart_type", "").lower()
        if ai_chart_type in self.chart_rules:
            # If AI recommendation matches a high-confidence rule-based rec, boost confidence
            if rule_based and ai_chart_type == rule_based[0][0]:
                confidence = min(confidence + 0.2, 1.0)
            # If no strong rule-based recommendation, use AI recommendation
            elif not rule_based or confidence < 0.6:
                chart_type = ai_chart_type
                confidence = 0.7

        return {
            "chart_type": chart_type,
            "confidence": confidence,
            "explanation": ai_recommendation.get("explanation", ""),
            "alternatives": [rec[0] for rec in rule_based[1:3]]  # Top 2 alternatives
        }

    def _create_chart_configuration(
            self,
            recommendation: Dict[str, Any],
            data_analysis: Dict[str, Any],
            user_question: str
    ) -> Dict[str, Any]:
        """Create detailed chart configuration."""
        chart_type = recommendation["chart_type"]
        columns = data_analysis["columns"]

        # Determine axes
        x_axis, y_axis = self._determine_axes(chart_type, columns, user_question)

        # Generate title
        title = self._generate_chart_title(user_question, chart_type)

        # Determine color column
        color_column = self._determine_color_column(columns, chart_type)

        config = {
            "chart_type": chart_type,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "title": title,
            "color_column": color_column,
            "confidence": recommendation["confidence"],
            "explanation": recommendation["explanation"],
            "alternatives": recommendation.get("alternatives", []),
            "metadata": {
                "data_points": data_analysis["row_count"],
                "columns_analyzed": len(columns),
                "recommendation_source": "combined"
            }
        }

        return config

    def _determine_axes(
            self,
            chart_type: str,
            columns: List[Dict[str, Any]],
            user_question: str
    ) -> Tuple[str, str]:
        """Determine appropriate X and Y axes for the chart."""
        categorical_cols = [col for col in columns if col["is_categorical"]]
        numerical_cols = [col for col in columns if col["is_numeric"]]
        temporal_cols = [col for col in columns if col["is_temporal"]]

        # Default axes
        x_axis = columns[0]["name"] if columns else ""
        y_axis = columns[1]["name"] if len(columns) > 1 else columns[0]["name"]

        if chart_type in ["line", "area"]:
            # For time series charts, prefer temporal column as X-axis
            if temporal_cols:
                x_axis = temporal_cols[0]["name"]
                if numerical_cols:
                    y_axis = numerical_cols[0]["name"]

        elif chart_type == "bar":
            # For bar charts, prefer categorical X-axis and numerical Y-axis
            if categorical_cols and numerical_cols:
                x_axis = categorical_cols[0]["name"]
                y_axis = numerical_cols[0]["name"]

        elif chart_type == "pie":
            # For pie charts, use categorical for labels and numerical for values
            if categorical_cols and numerical_cols:
                x_axis = categorical_cols[0]["name"]
                y_axis = numerical_cols[0]["name"]

        elif chart_type == "scatter":
            # For scatter plots, use two numerical columns
            if len(numerical_cols) >= 2:
                x_axis = numerical_cols[0]["name"]
                y_axis = numerical_cols[1]["name"]

        return x_axis, y_axis

    def _generate_chart_title(self, user_question: str, chart_type: str) -> str:
        """Generate appropriate chart title."""
        # Extract key terms from question
        question_words = user_question.split()

        # Remove common stop words
        stop_words = {"what", "how", "show", "me", "the", "a", "an", "is", "are", "by", "of", "in"}
        key_words = [word for word in question_words if word.lower() not in stop_words]

        # Create title
        if len(key_words) > 0:
            title = " ".join(key_words[:6]).title()  # First 6 meaningful words
        else:
            title = f"{chart_type.title()} Chart"

        return title

    def _determine_color_column(self, columns: List[Dict[str, Any]], chart_type: str) -> Optional[str]:
        """Determine if a color column would be beneficial."""
        categorical_cols = [col for col in columns if col["is_categorical"]]

        # For charts that benefit from color grouping
        if chart_type in ["bar", "scatter", "line"] and len(categorical_cols) > 1:
            # Choose categorical column with reasonable number of unique values
            suitable_cols = [col for col in categorical_cols if col["unique_count"] <= 10]
            if suitable_cols:
                return suitable_cols[0]["name"]

        return None

    def _create_empty_chart_response(self) -> Dict[str, Any]:
        """Create response for empty data."""
        return {
            "chart_type": "bar",
            "x_axis": "",
            "y_axis": "",
            "title": "No Data Available",
            "confidence": 0.0,
            "explanation": "No data available to visualize",
            "error": "No data returned from query"
        }

    def _create_fallback_chart_response(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create fallback response when recommendation fails."""
        if not data:
            return self._create_empty_chart_response()

        columns = list(data[0].keys())
        return {
            "chart_type": "bar",
            "x_axis": columns[0] if columns else "",
            "y_axis": columns[1] if len(columns) > 1 else columns[0] if columns else "",
            "title": "Data Visualization",
            "confidence": 0.5,
            "explanation": "Default chart configuration",
            "metadata": {"data_points": len(data)}
        }