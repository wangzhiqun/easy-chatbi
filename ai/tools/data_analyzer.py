"""
Data Analyzer for ChatBI platform.
Analyzes query results to extract insights, patterns, and generate summaries.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import re
from collections import Counter

from utils.logger import get_logger
from utils.exceptions import DataProcessingException, ErrorCodes

logger = get_logger(__name__)


class DataAnalyzer:
    """
    Intelligent data analyzer that examines query results to provide
    insights, detect patterns, and generate business-relevant summaries.
    """

    def __init__(self):
        """Initialize data analyzer with analysis patterns."""

        # Statistical significance thresholds
        self.min_sample_size = 10
        self.correlation_threshold = 0.7
        self.outlier_threshold = 2.0  # Standard deviations

        # Pattern detection settings
        self.trend_min_points = 5
        self.seasonality_min_periods = 12
        self.anomaly_detection_window = 30

    async def analyze_results(
            self,
            sql_query: str,
            results: List[Dict[str, Any]],
            user_question: str,
            context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of query results.

        Args:
            sql_query: SQL query that generated the results
            results: Query result data
            user_question: Original user question
            context: Additional context for analysis

        Returns:
            Comprehensive analysis report
        """
        logger.info(f"Analyzing {len(results)} data points")

        try:
            if not results:
                return self._create_empty_analysis()

            # Convert to DataFrame for analysis
            df = pd.DataFrame(results)

            # Perform different types of analysis
            analysis = {
                "basic_stats": self._calculate_basic_statistics(df),
                "data_quality": self._assess_data_quality(df),
                "patterns": self._detect_patterns(df),
                "insights": self._generate_insights(df, user_question),
                "trends": self._analyze_trends(df),
                "correlations": self._find_correlations(df),
                "outliers": self._detect_outliers(df),
                "summary": self._generate_summary(df, user_question),
                "recommendations": self._generate_recommendations(df, sql_query, user_question)
            }

            logger.info("Data analysis completed successfully")
            return analysis

        except Exception as e:
            logger.error(f"Data analysis failed: {e}")
            return self._create_error_analysis(str(e))

    def _calculate_basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistical measures for the dataset."""
        stats = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "data_types": {},
            "numeric_stats": {},
            "categorical_stats": {}
        }

        # Analyze each column
        for column in df.columns:
            col_data = df[column]
            stats["data_types"][column] = str(col_data.dtype)

            if pd.api.types.is_numeric_dtype(col_data):
                # Numeric column statistics
                stats["numeric_stats"][column] = {
                    "count": col_data.count(),
                    "mean": float(col_data.mean()) if not col_data.empty else 0,
                    "median": float(col_data.median()) if not col_data.empty else 0,
                    "std": float(col_data.std()) if not col_data.empty else 0,
                    "min": float(col_data.min()) if not col_data.empty else 0,
                    "max": float(col_data.max()) if not col_data.empty else 0,
                    "null_count": col_data.isnull().sum()
                }
            else:
                # Categorical column statistics
                stats["categorical_stats"][column] = {
                    "unique_count": col_data.nunique(),
                    "most_common": col_data.value_counts().head(5).to_dict(),
                    "null_count": col_data.isnull().sum()
                }

        return stats

    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess the quality of the dataset."""
        quality = {
            "completeness": {},
            "consistency": {},
            "validity": {},
            "overall_score": 0
        }

        total_cells = len(df) * len(df.columns)
        null_cells = df.isnull().sum().sum()

        # Completeness assessment
        quality["completeness"] = {
            "total_cells": total_cells,
            "null_cells": int(null_cells),
            "completeness_ratio": 1 - (null_cells / total_cells) if total_cells > 0 else 0,
            "columns_with_nulls": df.columns[df.isnull().any()].tolist()
        }

        # Consistency assessment
        duplicate_rows = df.duplicated().sum()
        quality["consistency"] = {
            "duplicate_rows": int(duplicate_rows),
            "uniqueness_ratio": 1 - (duplicate_rows / len(df)) if len(df) > 0 else 0
        }

        # Validity assessment (basic checks)
        validity_issues = []
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                if (df[column] < 0).any() and 'count' in column.lower():
                    validity_issues.append(f"Negative values in count column: {column}")

        quality["validity"] = {
            "issues": validity_issues,
            "valid_ratio": max(0, 1 - len(validity_issues) / len(df.columns))
        }

        # Calculate overall quality score
        quality["overall_score"] = (
                quality["completeness"]["completeness_ratio"] * 0.4 +
                quality["consistency"]["uniqueness_ratio"] * 0.3 +
                quality["validity"]["valid_ratio"] * 0.3
        )

        return quality

    def _detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect patterns in the data."""
        patterns = {
            "time_series": False,
            "seasonal": False,
            "categorical_distribution": {},
            "numeric_distributions": {},
            "data_skewness": {}
        }

        # Check for time series data
        date_columns = []
        for column in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[column]) or self._is_date_column(df[column]):
                date_columns.append(column)
                patterns["time_series"] = True

        # Analyze categorical distributions
        for column in df.columns:
            if not pd.api.types.is_numeric_dtype(df[column]):
                value_counts = df[column].value_counts()
                if len(value_counts) <= 20:  # Reasonable number of categories
                    patterns["categorical_distribution"][column] = {
                        "entropy": self._calculate_entropy(value_counts),
                        "gini_coefficient": self._calculate_gini(value_counts),
                        "top_categories": value_counts.head(5).to_dict()
                    }

        # Analyze numeric distributions
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]) and not df[column].empty:
                col_data = df[column].dropna()
                if len(col_data) > 0:
                    patterns["numeric_distributions"][column] = {
                        "distribution_type": self._identify_distribution(col_data),
                        "skewness": float(col_data.skew()),
                        "kurtosis": float(col_data.kurtosis())
                    }
                    patterns["data_skewness"][column] = float(col_data.skew())

        return patterns

    def _generate_insights(self, df: pd.DataFrame, user_question: str) -> List[Dict[str, Any]]:
        """Generate business insights from the data."""
        insights = []

        # Insight 1: Data volume insight
        row_count = len(df)
        if row_count > 1000:
            insights.append({
                "type": "volume",
                "message": f"Large dataset with {row_count:,} records provides robust statistical foundation",
                "confidence": 0.9
            })
        elif row_count < 10:
            insights.append({
                "type": "volume",
                "message": f"Small dataset ({row_count} records) may not be representative",
                "confidence": 0.8
            })

        # Insight 2: Missing data insights
        missing_data = df.isnull().sum()
        critical_missing = missing_data[missing_data > len(df) * 0.3]
        if not critical_missing.empty:
            insights.append({
                "type": "data_quality",
                "message": f"Significant missing data in columns: {', '.join(critical_missing.index)}",
                "confidence": 0.95
            })

        # Insight 3: Numeric insights
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]) and not df[column].empty:
                col_data = df[column].dropna()
                if len(col_data) > 0:
                    # Check for extreme values
                    q1, q3 = col_data.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    outliers = col_data[(col_data < q1 - 1.5 * iqr) | (col_data > q3 + 1.5 * iqr)]

                    if len(outliers) > 0:
                        insights.append({
                            "type": "outliers",
                            "message": f"Found {len(outliers)} potential outliers in {column}",
                            "confidence": 0.8
                        })

                    # Check for interesting patterns
                    if col_data.std() / col_data.mean() < 0.1:  # Low coefficient of variation
                        insights.append({
                            "type": "consistency",
                            "message": f"{column} shows very consistent values (low variation)",
                            "confidence": 0.7
                        })

        # Insight 4: Question-specific insights
        question_lower = user_question.lower()
        if 'trend' in question_lower or 'time' in question_lower:
            date_columns = [col for col in df.columns if self._is_date_column(df[col])]
            if date_columns:
                insights.append({
                    "type": "temporal",
                    "message": f"Time series data available in {', '.join(date_columns)}",
                    "confidence": 0.9
                })

        return insights

    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in time series data."""
        trends = {
            "has_time_data": False,
            "time_columns": [],
            "trend_analysis": {}
        }

        # Identify time columns
        for column in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[column]) or self._is_date_column(df[column]):
                trends["time_columns"].append(column)
                trends["has_time_data"] = True

        # Analyze trends for each time column
        for time_col in trends["time_columns"]:
            if len(df) >= self.trend_min_points:
                try:
                    # Convert to datetime if needed
                    time_data = pd.to_datetime(df[time_col])

                    # Find numeric columns to analyze trends
                    for num_col in df.columns:
                        if pd.api.types.is_numeric_dtype(df[num_col]):
                            trend_data = df[[time_col, num_col]].sort_values(time_col)
                            trend_analysis = self._calculate_trend(trend_data[num_col])

                            trends["trend_analysis"][f"{time_col}_{num_col}"] = trend_analysis

                except Exception as e:
                    logger.warning(f"Trend analysis failed for {time_col}: {e}")

        return trends

    def _find_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Find correlations between numeric variables."""
        correlations = {
            "correlation_matrix": {},
            "strong_correlations": [],
            "correlation_insights": []
        }

        # Get numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) >= 2:
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()
            correlations["correlation_matrix"] = corr_matrix.to_dict()

            # Find strong correlations
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]

                    if abs(corr_value) >= self.correlation_threshold:
                        correlations["strong_correlations"].append({
                            "variable1": col1,
                            "variable2": col2,
                            "correlation": float(corr_value),
                            "strength": "strong" if abs(corr_value) >= 0.8 else "moderate"
                        })

        return correlations

    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in numeric data."""
        outliers = {
            "outlier_columns": {},
            "outlier_counts": {},
            "outlier_summary": {}
        }

        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]) and not df[column].empty:
                col_data = df[column].dropna()

                if len(col_data) > 0:
                    # Z-score method
                    z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                    z_outliers = col_data[z_scores > self.outlier_threshold]

                    # IQR method
                    q1, q3 = col_data.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    iqr_outliers = col_data[(col_data < q1 - 1.5 * iqr) | (col_data > q3 + 1.5 * iqr)]

                    outliers["outlier_columns"][column] = {
                        "z_score_outliers": len(z_outliers),
                        "iqr_outliers": len(iqr_outliers),
                        "outlier_values": iqr_outliers.tolist()[:10]  # First 10 outliers
                    }

                    outliers["outlier_counts"][column] = len(iqr_outliers)

        return outliers

    def _generate_summary(self, df: pd.DataFrame, user_question: str) -> str:
        """Generate a natural language summary of the analysis."""
        row_count = len(df)
        col_count = len(df.columns)

        summary_parts = []

        # Basic data description
        summary_parts.append(f"The query returned {row_count:,} records with {col_count} columns.")

        # Data completeness
        null_percentage = (df.isnull().sum().sum() / (row_count * col_count)) * 100
        if null_percentage > 10:
            summary_parts.append(f"Data shows {null_percentage:.1f}% missing values.")
        elif null_percentage < 1:
            summary_parts.append("Data is highly complete with minimal missing values.")

        # Numeric data insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary_parts.append(f"Found {len(numeric_cols)} numeric columns for quantitative analysis.")

        # Categorical data insights
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(categorical_cols) > 0:
            summary_parts.append(f"Found {len(categorical_cols)} categorical columns for segmentation.")

        return " ".join(summary_parts)

    def _generate_recommendations(
            self,
            df: pd.DataFrame,
            sql_query: str,
            user_question: str
    ) -> List[str]:
        """Generate recommendations for further analysis."""
        recommendations = []

        # Data size recommendations
        if len(df) > 1000:
            recommendations.append("Consider filtering data for more focused analysis")
        elif len(df) < 10:
            recommendations.append("Consider expanding search criteria for more comprehensive analysis")

        # Missing data recommendations
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            recommendations.append(f"Investigate missing data in: {', '.join(missing_cols)}")

        # Analysis recommendations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            recommendations.append("Explore correlations between numeric variables")

        # Time series recommendations
        if any(self._is_date_column(df[col]) for col in df.columns):
            recommendations.append("Consider time-based trend analysis")

        # Visualization recommendations
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            recommendations.append("Create visualizations to show relationships between categories and metrics")

        return recommendations

    # Helper methods

    def _is_date_column(self, series: pd.Series) -> bool:
        """Check if a column contains date values."""
        if series.empty:
            return False

        sample = series.dropna().head(10)
        date_like_count = 0

        for value in sample:
            if isinstance(value, str):
                # Check for common date patterns
                date_patterns = [
                    r'\d{4}-\d{2}-\d{2}',
                    r'\d{2}/\d{2}/\d{4}',
                    r'\d{4}/\d{2}/\d{2}'
                ]
                if any(re.match(pattern, value) for pattern in date_patterns):
                    date_like_count += 1

        return date_like_count / len(sample) > 0.5

    def _calculate_entropy(self, value_counts: pd.Series) -> float:
        """Calculate entropy of categorical distribution."""
        probabilities = value_counts / value_counts.sum()
        return -sum(p * np.log2(p) for p in probabilities if p > 0)

    def _calculate_gini(self, value_counts: pd.Series) -> float:
        """Calculate Gini coefficient for categorical distribution."""
        sorted_values = np.sort(value_counts.values)
        n = len(sorted_values)
        cumulative = np.cumsum(sorted_values)
        return (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(sorted_values, 1))) / (n * cumulative[-1])

    def _identify_distribution(self, data: pd.Series) -> str:
        """Identify the likely statistical distribution of numeric data."""
        if len(data) < 10:
            return "insufficient_data"

        # Simple distribution identification based on skewness and kurtosis
        skewness = data.skew()
        kurtosis = data.kurtosis()

        if abs(skewness) < 0.5 and abs(kurtosis) < 1:
            return "normal"
        elif skewness > 1:
            return "right_skewed"
        elif skewness < -1:
            return "left_skewed"
        elif kurtosis > 2:
            return "heavy_tailed"
        else:
            return "unknown"

    def _calculate_trend(self, data: pd.Series) -> Dict[str, Any]:
        """Calculate trend statistics for time series data."""
        if len(data) < 2:
            return {"trend": "insufficient_data"}

        # Simple linear trend calculation
        x = np.arange(len(data))
        y = data.values

        # Calculate slope using least squares
        slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)

        # Determine trend direction
        if abs(slope) < 0.01:
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"

        return {
            "trend": trend_direction,
            "slope": float(slope),
            "start_value": float(data.iloc[0]),
            "end_value": float(data.iloc[-1]),
            "change_percentage": ((data.iloc[-1] - data.iloc[0]) / data.iloc[0] * 100) if data.iloc[0] != 0 else 0
        }

    def _create_empty_analysis(self) -> Dict[str, Any]:
        """Create analysis result for empty dataset."""
        return {
            "basic_stats": {"row_count": 0, "column_count": 0},
            "data_quality": {"overall_score": 0},
            "patterns": {},
            "insights": [{"type": "empty", "message": "No data available for analysis", "confidence": 1.0}],
            "trends": {"has_time_data": False},
            "correlations": {},
            "outliers": {},
            "summary": "No data returned from query.",
            "recommendations": ["Modify query to return data", "Check data availability"]
        }

    def _create_error_analysis(self, error_message: str) -> Dict[str, Any]:
        """Create analysis result for error cases."""
        return {
            "basic_stats": {"row_count": 0, "column_count": 0},
            "data_quality": {"overall_score": 0},
            "patterns": {},
            "insights": [{"type": "error", "message": f"Analysis failed: {error_message}", "confidence": 1.0}],
            "trends": {"has_time_data": False},
            "correlations": {},
            "outliers": {},
            "summary": f"Analysis could not be completed: {error_message}",
            "recommendations": ["Check data format", "Verify query results"]
        }