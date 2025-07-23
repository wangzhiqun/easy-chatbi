"""
Usage example for ChatBI AI module
Demonstrates how to use LLM client, SQL agent, and Chart agent
"""

import asyncio
import pandas as pd
from ai.llm_client import LLMClient
from ai.sql_agent import SQLAgent
from ai.chart_agent import ChartAgent
from ai.tools.data_analyzer import DataAnalyzer


async def main():
    """Main example function"""

    # Initialize LLM client
    print("1. Initializing LLM Client...")
    llm_client = LLMClient(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key="your-openai-api-key"  # Set your API key
    )

    # Test basic LLM functionality
    print("2. Testing basic LLM functionality...")
    response = await llm_client.generate_response(
        prompt="Explain what SQL is in simple terms.",
        system_message="You are a helpful database expert."
    )
    print(f"LLM Response: {response[:100]}...")

    # Initialize SQL Agent
    print("3. Initializing SQL Agent...")
    database_schema = {
        "sales": {
            "columns": ["id", "product_name", "quantity", "price", "sale_date", "customer_id"]
        },
        "customers": {
            "columns": ["id", "name", "email", "city", "country"]
        },
        "products": {
            "columns": ["id", "name", "category", "price", "stock_quantity"]
        }
    }

    sql_agent = SQLAgent(llm_client, database_schema)

    # Generate SQL query
    print("4. Generating SQL query...")
    natural_query = "Show me the total sales amount by product category for this month"
    sql_result = await sql_agent.generate_sql(natural_query)

    print(f"Generated SQL: {sql_result['sql_query']}")
    print(f"Is Valid: {sql_result['is_valid']}")
    print(f"Complexity: {sql_result['estimated_complexity']}")

    # Explain the SQL
    if sql_result['is_valid']:
        print("5. Explaining SQL query...")
        explanation = await sql_agent.explain_sql(sql_result['sql_query'])
        print(f"Explanation: {explanation[:150]}...")

    # Initialize Chart Agent
    print("6. Initializing Chart Agent...")
    chart_agent = ChartAgent(llm_client)

    # Create sample data for chart recommendation
    print("7. Creating sample data...")
    sample_data = pd.DataFrame({
        'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'sales': [15000, 18000, 22000, 19000, 25000, 28000],
        'orders': [120, 145, 180, 155, 200, 220],
        'category': ['Electronics', 'Electronics', 'Clothing', 'Electronics', 'Clothing', 'Electronics']
    })

    # Analyze data
    print("8. Analyzing data characteristics...")
    data_analyzer = DataAnalyzer()
    data_analysis = data_analyzer.analyze_dataframe(sample_data)

    print(f"Data Info:")
    print(f"  - Rows: {data_analysis['num_rows']}")
    print(f"  - Columns: {data_analysis['num_columns']}")
    print(f"  - Numeric columns: {data_analysis['numeric_columns_count']}")
    print(f"  - Categorical columns: {data_analysis['categorical_columns_count']}")
    print(f"  - Has temporal data: {data_analysis['has_temporal_column']}")

    # Get chart recommendations
    print("9. Getting chart recommendations...")
    chart_recommendations = await chart_agent.recommend_charts(
        data=sample_data,
        user_intent="I want to see sales trends over time",
        max_recommendations=3
    )

    print("Chart Recommendations:")
    for i, rec in enumerate(chart_recommendations, 1):
        print(f"  {i}. {rec['chart_type']} (confidence: {rec.get('confidence', 0):.2f})")
        print(f"     Reason: {rec.get('reason', 'N/A')}")

    # Generate chart configuration
    if chart_recommendations:
        print("10. Generating chart configuration...")
        best_chart = chart_recommendations[0]
        chart_config = await chart_agent.generate_chart_config(
            data=sample_data,
            chart_type=best_chart['chart_type']
        )

        print(f"Chart Config for {best_chart['chart_type']}:")
        print(f"  - Title: {chart_config.get('title', 'N/A')}")
        print(f"  - Width: {chart_config.get('width', 'N/A')}")
        print(f"  - Height: {chart_config.get('height', 'N/A')}")

    # Data quality analysis
    print("11. Performing data quality analysis...")
    outliers = data_analyzer.detect_outliers(sample_data)
    print(f"Outliers detected: {outliers['outliers_found']}")

    if outliers['outliers_found']:
        print(f"Columns with outliers: {outliers['columns_with_outliers']}")

    # Correlation analysis
    print("12. Analyzing correlations...")
    correlations = data_analyzer.calculate_correlation_matrix(sample_data)

    if correlations.get('has_correlations'):
        strong_corr = correlations.get('strong_correlations', [])
        print(f"Strong correlations found: {len(strong_corr)}")
        for corr in strong_corr[:3]:  # Show first 3
            print(f"  - {corr['column1']} vs {corr['column2']}: {corr['correlation']:.3f}")

    print("13. AI Module demo completed!")


def sync_example():
    """Synchronous example for basic functionality"""

    print("Synchronous Example:")

    # Initialize components
    llm_client = LLMClient(api_key="your-openai-api-key")
    sql_agent = SQLAgent(llm_client)
    data_analyzer = DataAnalyzer()

    # Test SQL generation (sync)
    natural_query = "Get all customers from New York"
    sql_result = sql_agent.generate_sql_sync(natural_query)

    print(f"SQL Query: {sql_result['sql_query']}")
    print(f"Valid: {sql_result['is_valid']}")

    # Test data analysis
    sample_data = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'salary': [50000, 60000, 70000]
    })

    analysis = data_analyzer.analyze_dataframe(sample_data)
    print(f"Data analysis: {analysis['num_rows']} rows, {analysis['num_columns']} columns")


def data_transformation_example():
    """Example of data transformation suggestions"""

    print("Data Transformation Example:")

    # Create messy data
    messy_data = pd.DataFrame({
        'date_string': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'numeric_string': ['100', '200', '300'],
        'mixed_data': ['A', 'B', 'C'],
        'high_cardinality': [f'item_{i}' for i in range(100)],
        'missing_values': [1, None, 3]
    })

    data_analyzer = DataAnalyzer()
    suggestions = data_analyzer.suggest_data_transformations(messy_data)

    print("Transformation suggestions:")
    for suggestion in suggestions:
        print(f"Column: {suggestion['column']}")
        for s in suggestion['suggestions']:
            print(f"  - {s['type']}: {s['reason']}")


if __name__ == "__main__":
    # Run async example
    print("=== Async AI Module Example ===")
    asyncio.run(main())

    print("\n=== Sync Example ===")
    sync_example()

    print("\n=== Data Transformation Example ===")
    data_transformation_example()