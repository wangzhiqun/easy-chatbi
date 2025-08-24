import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.utils import api_client, helpers

st.set_page_config(
    page_title="ChatBI - Data Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'query_result' not in st.session_state:
    st.session_state.query_result = None
if 'query_content' not in st.session_state:
    st.session_state.query_content = None


def main():
    with st.sidebar:
        st.title("Easy-ChatBI")

        st.markdown("""
                        **Easy-ChatBI** 是一个开源的简易AI驱动数据分析平台
                        
                        🔗 **GitHub:** [easy-chatbi](https://github.com/your-username/easy-chatbi) **♥️Star**
                    """)

        st.markdown("---")

        st.subheader("Navigation")
        page = st.radio(
            "Select Page",
            # ["💬 Chat", "📊 Data Query", "📈 Visualization", "🔧 Tools"],
            ["💬 Chat", "📊 Data Query", "🔧 Tools"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        st.subheader("API Health")
        api_url = st.text_input(
            "API URL",
            value="http://localhost:8000",
            help="Base URL for the ChatBI API"
        )

        if st.button("Test Connection"):
            if api_client.test_connection(api_url):
                st.success("✅ Connected to API")
            else:
                st.error("❌ Failed to connect to API")

        st.markdown("---")

        st.info(
            "**ChatBI v1.0.0**\n\n"
            "Data Intelligence Platform powered by AI"
        )

    if page == "💬 Chat":
        show_chat_page()
    elif page == "📊 Data Query":
        show_data_query_page()
    # elif page == "📈 Visualization":
    #     show_visualization_page()
    elif page == "🔧 Tools":
        show_tools_page()


def send_message(message: str):
    try:
        st.session_state.messages.append({"role": "user", "content": message})

        with st.chat_message("user"):
            st.markdown(message)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = api_client.send_chat_message(
                    message,
                    st.session_state.conversation_id
                )

                if response:
                    st.session_state.conversation_id = response.get('conversation_id')

                    content = response.get('response', 'Sorry, I could not process your request.')
                    st.markdown(content)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": content,
                        "metadata": response.get('metadata')
                    })

                    if response.get('metadata', {}).get('result'):
                        st.session_state.query_result = response['metadata']['result']
                else:
                    st.error("Failed to get response from API")

        st.rerun()

    except Exception as e:
        st.error(f"处理消息时出错: {str(e)}")
        if st.session_state.messages and st.session_state.messages[-1]["content"] == message:
            st.session_state.messages.pop()


def show_chat_page():
    col1, col2 = st.columns([2, 1])

    with col1:

        st.title("💬 AI Chat")
        st.markdown("Ask questions about your data in natural language")

        chat_container = st.container()

        with chat_container:
            for i, message in enumerate(st.session_state.messages):

                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

                    metadata = message.get("metadata", {})
                    if metadata.get('has_chart', False):
                        helpers.render_chart_from_metadata(metadata, i)

                    # elif metadata.get('intent') == 'sql_query' and metadata.get('result'):
                    #     helpers.render_query_results(metadata['result'], i)

                    # Display metadata if available
                    # if "metadata" in message and message["metadata"]:
                    #     with st.expander("Details"):
                    #         st.json(message["metadata"])

        if prompt := st.chat_input("Ask me about your data..."):
            send_message(prompt)

    with col2:
        st.subheader("Chat Controls")

        if st.button("🆕 New Conversation"):
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.session_state.query_result = None
            st.rerun()

        if st.button("💾 Save Conversation"):
            if st.session_state.messages:
                helpers.save_conversation(
                    st.session_state.conversation_id,
                    st.session_state.messages
                )
                st.success("Conversation saved!")
            else:
                st.warning("No messages to save")

        if st.button("🗑️ Clear History"):
            st.session_state.messages = []
            st.rerun()


def show_data_query_page():
    st.title("📊 Data Query")
    st.markdown("Execute SQL queries and explore your data")

    st.subheader("SQL Query / Natural Language")

    with st.expander("💡 使用说明"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **SQL查询示例：**
            ```sql
            SELECT * FROM products 
            WHERE price > 100 
            LIMIT 10
            ```
            """)
        with col2:
            st.markdown("""
            **自然语言示例：**
            - 查询价格超过100元的商品
            - 统计每个地区的用户数量  
            - 找出销量最高的10个产品
            - 显示最近30天的订单信息
            """)

    query = st.text_area(
        "输入SQL查询语句或用自然语言描述您的需求",
        value=st.session_state.query_content,
        placeholder="""可以输入：
1. SQL查询：SELECT * FROM products LIMIT 10
2. 自然语言：查询所有产品信息，限制10条记录""",
        height=150,
    )

    if query != st.session_state.query_content:
        st.session_state.query_content = query

    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        if st.button("🌟 AI生成SQL", help="将自然语言转换为SQL查询"):
            if query.strip():
                generate_sql_from_natural_language(query)
            else:
                st.warning("请先输入查询描述")

    with col2:
        if st.button("▶️ 执行查询"):
            if query.strip():
                if is_sql_query(query):
                    execute_sql_query(query)
                else:
                    st.info("检测到自然语言，正在生成SQL...")
                    if generate_sql_from_natural_language(query, auto_execute=True):
                        pass
            else:
                st.warning("请输入查询语句")

    with col3:
        if st.button("✅ 验证SQL"):
            if query.strip():
                if is_sql_query(query):
                    validate_sql_query(query)
                else:
                    st.warning("请先生成SQL或输入有效的SQL语句")
            else:
                st.warning("请输入查询语句")

    with col4:
        if st.button("🗑️ 清除", type="primary"):
            st.session_state.query_content = ""
            st.rerun()

    if 'last_generated_explanation' in st.session_state and st.session_state.last_generated_explanation:
        with st.expander("📖 上次生成的SQL解释"):
            st.markdown(st.session_state.last_generated_explanation)

    if 'last_optimization_tips' in st.session_state and st.session_state.last_optimization_tips:
        with st.expander("⚡ 优化建议"):
            st.markdown(st.session_state.last_optimization_tips)

    show_query_results()


def is_sql_query(text: str) -> bool:
    text_lower = text.strip().lower()
    sql_keywords = ['select', 'insert', 'update', 'delete', 'create', 'drop', 'alter', 'show', 'describe']

    for keyword in sql_keywords:
        if text_lower.startswith(keyword):
            return True

    sql_patterns = ['from ', 'where ', 'select ', 'group by', 'order by', 'limit ']
    sql_pattern_count = sum(1 for pattern in sql_patterns if pattern in text_lower)

    return sql_pattern_count >= 2


def generate_sql_from_natural_language(description: str, auto_execute: bool = False) -> bool:
    include_explanation = st.session_state.get('include_explanation', True)
    with st.spinner("🤖 AI正在理解您的需求并生成SQL..."):
        try:
            response = api_client.generate_sql_from_description({
                'description': description,
                'include_explanation': include_explanation
            })

            st.warning(response)

            if response and response.get('status') == 'success':
                generated_sql = response.get('sql', '')

                st.session_state.query_content = generated_sql

                if include_explanation:
                    st.session_state.last_generated_explanation = response.get('explanation', '')

                st.success("✅ SQL生成成功！")

                st.code(generated_sql, language='sql')

                if auto_execute:
                    execute_sql_query(generated_sql)
                else:
                    st.rerun()

                return True

            else:
                error_msg = response.get('error', '未知错误') if response else 'API调用失败'
                st.error(f"❌ 生成SQL失败：{error_msg}")

                st.info("""
                **💡 改进建议：**
                - 请提供更具体的描述
                - 明确指定表名和字段名
                - 说明需要的时间范围和筛选条件
                - 可以参考山上方的查询模板
                """)
                return False

        except Exception as e:
            st.error(f"❌ 处理请求时出错：{str(e)}")
            return False


def execute_sql_query(query: str):
    if query:
        with st.spinner("正在执行查询..."):
            result = api_client.execute_query(query)

            if result and result.get('status') == 'success':
                st.session_state.query_result = result
                st.success(f"✅ 查询执行成功！返回 {result.get('row_count', 0)} 行数据")
            else:
                error_msg = result.get('error', '未知错误') if result else 'API调用失败'
                st.error(f"❌ 查询失败：{error_msg}")
    else:
        st.warning("⚠️ 请输入查询语句")


def validate_sql_query(query: str):
    if query:
        validation = api_client.validate_sql(query)
        if validation.get('valid'):
            st.success("✅ SQL语法正确且安全")
        else:
            st.error(f"❌ SQL验证失败：{validation.get('error')}")
    else:
        st.warning("⚠️ 请输入查询语句")


def show_query_results():
    if st.session_state.query_result:
        result = st.session_state.query_result

        st.markdown("---")
        st.subheader("📊 查询结果")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("行数", result.get('row_count', 0))
        with col2:
            st.metric("列数", len(result.get('columns', [])))
        with col3:
            if 'execution_time' in result:
                st.metric("执行时间", f"{result['execution_time']}ms")
        with col4:
            st.metric("状态", result.get('status', 'unknown'))

        if result.get('data'):
            df = pd.DataFrame(result['data'])

            col1, col2 = st.columns([3, 1])
            with col2:
                display_option = st.radio(
                    "显示格式",
                    ["表格", "原始数据"],
                    horizontal=True
                )

            if display_option == "表格":
                st.dataframe(df, use_container_width=True)
            else:
                st.json(result['data'])

            st.subheader("📥 导出和可视化")
            col1, col2 = st.columns(2)

            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 下载 CSV",
                    data=csv,
                    file_name=f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            with col2:
                json_str = df.to_json(orient='records', force_ascii=False)
                st.download_button(
                    label="📥 下载 JSON",
                    data=json_str,
                    file_name=f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )


def show_visualization_page():
    st.title("📈 Data Visualization")
    st.markdown("Create interactive charts and dashboards")

    data_source = st.radio(
        "Data Source",
        ["Query Result", "Upload File", "Sample Data"],
        horizontal=True
    )

    df = None

    if data_source == "Query Result":
        if st.session_state.query_result and st.session_state.query_result.get('data'):
            df = pd.DataFrame(st.session_state.query_result['data'])
            st.success(f"Using query result with {len(df)} rows")
        else:
            st.warning("No query result available. Please execute a query first.")

    elif data_source == "Upload File":
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'json']
        )

        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)

                st.success(f"Loaded {len(df)} rows from {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

    elif data_source == "Sample Data":
        df = helpers.load_sample_data()
        st.success(f"Loaded sample data with {len(df)} rows")

    if df is not None:
        st.markdown("---")

        col1, col2 = st.columns([2, 1])

        with col1:
            chart_container = st.container()

        with col2:
            st.subheader("Chart Configuration")

            chart_type = st.selectbox(
                "Chart Type",
                ["Auto", "Line", "Bar", "Scatter", "Pie", "Heatmap", "Box", "Histogram"]
            )

            if chart_type != "Auto":
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                all_cols = df.columns.tolist()

                if chart_type in ["Line", "Bar", "Scatter"]:
                    x_axis = st.selectbox("X Axis", all_cols)
                    y_axis = st.selectbox("Y Axis", numeric_cols)

                    color_by = st.selectbox(
                        "Color By (optional)",
                        ["None"] + categorical_cols
                    )
                elif chart_type == "Pie":
                    values = st.selectbox("Values", numeric_cols)
                    names = st.selectbox("Names", categorical_cols)
                elif chart_type in ["Histogram", "Box"]:
                    column = st.selectbox("Column", numeric_cols)

            if st.button("🎨 Create Chart", type="primary"):
                with st.spinner("Creating chart..."):
                    if chart_type == "Auto":
                        chart_config = api_client.create_chart(
                            df.to_dict('records'),
                            chart_type="auto"
                        )
                    else:
                        options = {}
                        if chart_type in ["Line", "Bar", "Scatter"]:
                            options['x_axis'] = x_axis
                            options['y_axis'] = y_axis
                            if color_by != "None":
                                options['color'] = color_by
                        elif chart_type == "Pie":
                            options['values'] = values
                            options['names'] = names
                        elif chart_type in ["Histogram", "Box"]:
                            options['column'] = column

                        chart_config = api_client.create_chart(
                            df.head(1000).to_dict('records'),
                            chart_type=chart_type.lower(),
                            options=options
                        )

                    if chart_config and chart_config.get('status') == 'success':
                        with chart_container:
                            helpers.render_chart(chart_config['config'], df)

                        st.success("Chart created successfully!")
                    else:
                        st.error("Failed to create chart")

        with st.expander("📋 Data Preview"):
            st.dataframe(df.head(100), use_container_width=True)

        with st.expander("📊 Data Statistics"):
            st.write(df.describe())


def show_tools_page():
    st.title("🔧 Tools & Utilities")
    st.markdown("Advanced tools for data management and analysis")

    tool_type = st.selectbox(
        "Select Tool",
        ["Database Schema", "MCP Tools", "Knowledge Base", "Cache Management", "Security"]
    )

    st.markdown("---")

    if tool_type == "Database Schema":
        show_schema_tool()
    elif tool_type == "MCP Tools":
        show_mcp_tools()
    elif tool_type == "Knowledge Base":
        show_knowledge_base()
    elif tool_type == "Cache Management":
        show_cache_management()
    elif tool_type == "Security":
        show_security_tools()


def show_schema_tool():
    st.subheader("Database Schema")

    if st.button("🔄 Refresh Schema"):
        with st.spinner("Loading schema..."):
            schema = api_client.get_schema()

            if schema:
                st.session_state.schema = schema
                st.success("Schema loaded successfully")
            else:
                st.error("Failed to load schema")

    if 'schema' in st.session_state:
        schema = st.session_state.schema

        tables = schema.get('tables', {})

        selected_table = st.selectbox(
            "Select Table",
            list(tables.keys())
        )

        if selected_table:
            table_info = tables[selected_table]

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Row Count", table_info.get('row_count', 'N/A'))

            with col2:
                st.metric("Column Count", len(table_info.get('columns', [])))

            st.subheader("Columns")

            columns_df = pd.DataFrame(table_info.get('columns', []))
            if not columns_df.empty:
                st.dataframe(
                    columns_df[['COLUMN_NAME', 'DATA_TYPE', 'IS_NULLABLE', 'COLUMN_KEY']],
                    use_container_width=True
                )

            if st.button(f"📋 Get Sample Data from {selected_table}"):
                with st.spinner("Loading sample data..."):
                    sample = api_client.get_table_info(selected_table)
                    if sample and 'sample_data' in sample:
                        st.subheader("Sample Data")
                        st.dataframe(pd.DataFrame(sample['sample_data']))


def show_mcp_tools():
    st.subheader("MCP Tools")

    tools = api_client.list_mcp_tools()

    if tools:
        selected_tool = st.selectbox(
            "Select Tool",
            [tool['name'] for tool in tools]
        )

        tool_info = next((t for t in tools if t['name'] == selected_tool), None)

        if tool_info:
            st.write(f"**Description:** {tool_info.get('description', 'N/A')}")

            st.subheader("Parameters")
            params = {}

            if selected_tool == "query_database":
                params['query'] = st.text_area("SQL Query")
                params['database'] = st.text_input("Database (optional)")

            elif selected_tool == "generate_sql":
                params['question'] = st.text_input("Natural Language Question")

            elif selected_tool == "analyze_data":
                params['analysis_type'] = st.selectbox(
                    "Analysis Type",
                    ["comprehensive", "correlation", "anomaly", "trend"]
                )

            if st.button("▶️ Execute Tool"):
                with st.spinner("Executing..."):
                    result = api_client.execute_mcp_tool(selected_tool, params)

                    if result:
                        st.success("Tool executed successfully!")
                        st.json(result)
                    else:
                        st.error("Tool execution failed")


def show_knowledge_base():
    st.subheader("Knowledge Base")

    tab1, tab2 = st.tabs(["Search", "Add Knowledge"])

    with tab1:
        query = st.text_input("Search Query")

        if st.button("🔍 Search"):
            if query:
                with st.spinner("Searching..."):
                    results = api_client.search_knowledge(query)

                    if results:
                        st.success(f"Found {len(results)} results")

                        for result in results:
                            with st.expander(f"📄 {result.get('title', 'Untitled')}"):
                                st.write(result.get('content', ''))
                                st.caption(f"Score: {result.get('score', 0):.3f}")
                    else:
                        st.info("No results found")

    with tab2:
        title = st.text_input("Title")
        content = st.text_area("Content", height=200)
        category = st.selectbox("Category", ["General", "SQL", "Analysis", "Visualization"])
        tags = st.text_input("Tags (comma-separated)")

        if st.button("💾 Add to Knowledge Base"):
            if title and content:
                success = api_client.add_knowledge(
                    title,
                    content,
                    category,
                    tags.split(',') if tags else []
                )

                if success:
                    st.success("Knowledge added successfully!")
                else:
                    st.error("Failed to add knowledge")
            else:
                st.warning("Please provide title and content")


def show_cache_management():
    st.subheader("Cache Management")

    stats = api_client.get_cache_stats()

    if stats:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Status", "🟢 Connected" if stats.get('connected') else "🔴 Disconnected")

        with col2:
            st.metric("Memory Usage", stats.get('used_memory', 'N/A'))

        with col3:
            st.metric("Hit Rate", f"{stats.get('hit_rate', 0):.1f}%")

        with col4:
            st.metric("Total Commands", stats.get('total_commands_processed', 0))

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🗑️ Clear Query Cache"):
            if api_client.clear_cache("queries"):
                st.success("Query cache cleared")
            else:
                st.error("Failed to clear cache")

    with col2:
        if st.button("🗑️ Clear Analysis Cache"):
            if api_client.clear_cache("analysis"):
                st.success("Analysis cache cleared")
            else:
                st.error("Failed to clear cache")

    with col3:
        if st.button("🗑️ Clear All Cache"):
            if api_client.clear_cache("all"):
                st.success("All cache cleared")
            else:
                st.error("Failed to clear cache")


def show_security_tools():
    st.subheader("Security Tools")

    tab1, tab2 = st.tabs(["SQL Validator", "API Key Management"])

    with tab1:
        st.write("Validate SQL queries for safety and syntax")

        query = st.text_area("SQL Query to Validate", height=150)

        if st.button("✅ Validate"):
            if query:
                result = api_client.validate_sql(query)

                if result.get('valid'):
                    st.success("✅ Query is valid and safe to execute")
                else:
                    st.error(f"❌ Invalid query: {result.get('error')}")

                    st.info(
                        "**Tips for safe queries:**\n"
                        "- Use only SELECT statements\n"
                        "- Avoid DROP, DELETE, TRUNCATE operations\n"
                        "- Check for balanced quotes and parentheses\n"
                        "- Limit query length to under 10000 characters"
                    )

    with tab2:
        st.write("Generate and manage API keys")

        if st.button("🔑 Generate New API Key"):
            api_key = api_client.generate_api_key()
            if api_key:
                st.success("New API key generated!")
                st.code(api_key)
                st.warning("⚠️ Save this key securely. It won't be shown again.")
            else:
                st.error("Failed to generate API key")


if __name__ == "__main__":
    main()
