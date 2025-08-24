from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from utils import logger, AIError
from ..llm_client import LLMClient
from ..prompts import PromptTemplates
from ..tools import AnalysisTool


class AnalysisAgent:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self.analysis_tool = AnalysisTool()
        logger.info("Initialized Analysis Agent")

    def analyze_data(
            self,
            df: pd.DataFrame,
            analysis_request: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            statistics = self.analysis_tool.generate_statistics(df)

            patterns = self.analysis_tool.detect_patterns(df)

            data_overview = self._create_data_overview(df)
            statistics_text = self._format_statistics(statistics)

            system_prompt, user_prompt = PromptTemplates.get_analysis_prompt(
                data_overview=data_overview,
                statistics=statistics_text,
                analysis_request=analysis_request or "请进行全面的数据分析"
            )

            llm_analysis = self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.6
            )

            analysis_results = {
                'summary': self._extract_summary(llm_analysis),
                'statistics': statistics,
                'patterns': patterns,
                'insights': self._extract_insights(llm_analysis),
                'recommendations': self._extract_recommendations(llm_analysis),
                'visualizations': self.analysis_tool.suggest_visualizations(df),
                'full_analysis': llm_analysis
            }

            logger.info("Data analysis completed successfully")
            return analysis_results

        except Exception as e:
            logger.error(f"Data analysis failed: {str(e)}")
            raise AIError(f"Data analysis failed: {str(e)}")

    def generate_report(
            self,
            analysis_results: Dict[str, Any],
            format: str = 'markdown'
    ) -> str:
        try:
            if format == 'markdown':
                report = self._generate_markdown_report(analysis_results)
            elif format == 'html':
                report = self._generate_html_report(analysis_results)
            elif format == 'text':
                report = self._generate_text_report(analysis_results)
            else:
                raise ValueError(f"Unsupported report format: {format}")

            logger.info(f"Generated {format} report")
            return report

        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            raise AIError(f"Report generation failed: {str(e)}")

    def find_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            numeric_df = df.select_dtypes(include=['number'])

            if len(numeric_df.columns) < 2:
                return {'message': 'Not enough numeric columns for correlation analysis'}

            corr_matrix = numeric_df.corr()

            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        strong_correlations.append({
                            'column1': corr_matrix.columns[i],
                            'column2': corr_matrix.columns[j],
                            'correlation': round(corr_value, 3),
                            'strength': 'strong' if abs(corr_value) > 0.9 else 'moderate'
                        })

            strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

            return {
                'correlation_matrix': corr_matrix.to_dict(),
                'strong_correlations': strong_correlations,
                'interpretation': self._interpret_correlations(strong_correlations)
            }

        except Exception as e:
            logger.error(f"Correlation analysis failed: {str(e)}")
            raise AIError(f"Correlation analysis failed: {str(e)}")

    def detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            anomalies = {}
            numeric_cols = df.select_dtypes(include=['number']).columns

            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

                if len(outliers) > 0:
                    anomalies[col] = {
                        'count': len(outliers),
                        'percentage': round(len(outliers) / len(df) * 100, 2),
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'outlier_indices': outliers.index.tolist()[:10]
                    }

            if len(numeric_cols) > 1:
                from scipy import stats
                z_scores = np.abs(stats.zscore(df[numeric_cols]))
                threshold = 3
                multivariate_anomalies = (z_scores > threshold).any(axis=1)

                anomalies['multivariate'] = {
                    'count': multivariate_anomalies.sum(),
                    'percentage': round(multivariate_anomalies.sum() / len(df) * 100, 2),
                    'indices': df[multivariate_anomalies].index.tolist()[:10]
                }

            return {
                'anomalies': anomalies,
                'summary': self._summarize_anomalies(anomalies),
                'recommendations': self._anomaly_recommendations(anomalies)
            }

        except Exception as e:
            logger.error(f"Anomaly detection failed: {str(e)}")
            raise AIError(f"Anomaly detection failed: {str(e)}")

    def trend_analysis(self, df: pd.DataFrame, time_column: Optional[str] = None) -> Dict[str, Any]:
        try:
            if not time_column:
                date_cols = [col for col in df.columns
                             if 'date' in col.lower() or 'time' in col.lower()]
                if date_cols:
                    time_column = date_cols[0]
                else:
                    return {'error': 'No time column found for trend analysis'}

            if time_column in df.columns:
                df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
                df = df.sort_values(time_column)

            trends = {}
            numeric_cols = df.select_dtypes(include=['number']).columns

            for col in numeric_cols:
                values = df[col].values
                x = np.arange(len(values))

                coefficients = np.polyfit(x, values, 1)
                trend_direction = 'increasing' if coefficients[0] > 0 else 'decreasing'

                ma_7 = df[col].rolling(window=7, min_periods=1).mean()
                ma_30 = df[col].rolling(window=30, min_periods=1).mean()

                trends[col] = {
                    'direction': trend_direction,
                    'slope': float(coefficients[0]),
                    'recent_change': float((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0,
                    'volatility': float(df[col].std()),
                    'moving_avg_7': ma_7.iloc[-1] if len(ma_7) > 0 else None,
                    'moving_avg_30': ma_30.iloc[-1] if len(ma_30) > 0 else None
                }

            return {
                'trends': trends,
                'interpretation': self._interpret_trends(trends),
                'forecast_suggestion': 'Consider using advanced time series models for forecasting'
            }

        except Exception as e:
            logger.error(f"Trend analysis failed: {str(e)}")
            raise AIError(f"Trend analysis failed: {str(e)}")

    def _create_data_overview(self, df: pd.DataFrame) -> str:
        overview = f"""
数据集概览：
- 总行数：{len(df)}
- 总列数：{len(df.columns)}
- 列名：{', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}
- 数据类型分布：
  * 数值型：{len(df.select_dtypes(include=['number']).columns)}列
  * 文本型：{len(df.select_dtypes(include=['object']).columns)}列
  * 日期型：{len(df.select_dtypes(include=['datetime']).columns)}列
- 缺失值：{df.isnull().sum().sum()}个
"""
        return overview

    def _format_statistics(self, statistics: Dict[str, Any]) -> str:
        stats_text = "统计信息：\n"

        if 'numeric_summary' in statistics:
            stats_text += "\n数值列统计：\n"
            for col, stats in statistics['numeric_summary'].items():
                if isinstance(stats, dict) and 'mean' in stats:
                    stats_text += f"  {col}: 均值={stats['mean']:.2f}, 标准差={stats.get('std', 0):.2f}\n"

        if 'categorical_summary' in statistics:
            stats_text += "\n分类列统计：\n"
            for col, stats in statistics['categorical_summary'].items():
                stats_text += f"  {col}: 唯一值={stats['unique_count']}, 最频繁={stats['top_value']}\n"

        return stats_text

    def _extract_summary(self, analysis: str) -> str:
        lines = analysis.split('\n')
        summary_lines = []
        in_summary = False

        for line in lines:
            if '总结' in line or '概要' in line or 'summary' in line.lower():
                in_summary = True
                continue
            elif in_summary and line.strip():
                summary_lines.append(line.strip())
                if len(summary_lines) >= 3:
                    break

        return ' '.join(summary_lines) if summary_lines else analysis[:200]

    def _extract_insights(self, analysis: str) -> List[str]:
        insights = []
        lines = analysis.split('\n')

        for line in lines:
            if any(keyword in line for keyword in ['发现', '显示', '表明', '趋势', '模式']):
                insight = line.strip()
                if insight and len(insight) > 10:
                    insights.append(insight)

        return insights[:5]

    def _extract_recommendations(self, analysis: str) -> List[str]:
        recommendations = []
        lines = analysis.split('\n')

        for line in lines:
            if any(keyword in line for keyword in ['建议', '应该', '可以', '需要', '推荐']):
                rec = line.strip()
                if rec and len(rec) > 10:
                    recommendations.append(rec)

        return recommendations[:3]

    def _interpret_correlations(self, correlations: List[Dict]) -> str:
        if not correlations:
            return "没有发现强相关性"

        interpretation = "相关性分析发现：\n"
        for corr in correlations[:3]:
            interpretation += f"- {corr['column1']} 与 {corr['column2']} "
            if corr['correlation'] > 0:
                interpretation += f"正相关 ({corr['correlation']})\n"
            else:
                interpretation += f"负相关 ({corr['correlation']})\n"

        return interpretation

    def _summarize_anomalies(self, anomalies: Dict) -> str:
        if not anomalies:
            return "未检测到异常值"

        total_anomalies = sum(a['count'] for a in anomalies.values() if isinstance(a, dict) and 'count' in a)
        summary = f"共检测到 {total_anomalies} 个异常值，分布在 {len(anomalies)} 个特征中"

        return summary

    def _anomaly_recommendations(self, anomalies: Dict) -> List[str]:
        recommendations = []

        if anomalies:
            recommendations.append("建议进一步调查异常值的原因")
            recommendations.append("考虑使用稳健的统计方法处理异常值")
            recommendations.append("可以选择删除、替换或保留异常值，取决于业务场景")

        return recommendations

    def _interpret_trends(self, trends: Dict) -> str:
        if not trends:
            return "无趋势数据"

        interpretation = "趋势分析显示：\n"
        for col, trend in list(trends.items())[:3]:
            interpretation += f"- {col}: {trend['direction']}趋势，"
            interpretation += f"变化率 {trend['recent_change']:.2f}%\n"

        return interpretation

    def _generate_markdown_report(self, analysis_results: Dict[str, Any]) -> str:
        report = f"""# 数据分析报告

## 执行摘要
{analysis_results.get('summary', 'N/A')}

## 关键发现
"""
        for i, insight in enumerate(analysis_results.get('insights', []), 1):
            report += f"{i}. {insight}\n"

        report += """
## 统计概览
"""
        stats = analysis_results.get('statistics', {})
        report += f"- 数据行数: {stats.get('row_count', 'N/A')}\n"
        report += f"- 数据列数: {stats.get('column_count', 'N/A')}\n"

        report += """
## 建议
"""
        for i, rec in enumerate(analysis_results.get('recommendations', []), 1):
            report += f"{i}. {rec}\n"

        report += """
## 推荐可视化
"""
        for viz in analysis_results.get('visualizations', []):
            report += f"- **{viz['type']}**: {viz.get('description', '')}\n"

        return report

    def _generate_html_report(self, analysis_results: Dict[str, Any]) -> str:
        md_report = self._generate_markdown_report(analysis_results)

        html_report = f"""<!DOCTYPE html>
<html>
<head>
    <title>Data Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; }}
        ul {{ line-height: 1.6; }}
    </style>
</head>
<body>
    {md_report.replace('#', '').replace('*', '')}
</body>
</html>"""

        return html_report

    def _generate_text_report(self, analysis_results: Dict[str, Any]) -> str:
        md_report = self._generate_markdown_report(analysis_results)
        text_report = md_report.replace('#', '').replace('*', '').replace('-', '•')
        return text_report
