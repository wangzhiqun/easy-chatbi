# ChatBI - Intelligent Data Analysis Platform

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.20-orange.svg)](https://langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Overview

ChatBI is an intelligent data analysis platform that combines natural language processing with business intelligence. Users can interact with their data using natural language queries, automatically generate SQL queries, create visualizations, and gain insights through an intuitive chat interface.

### ✨ Key Features

- **🤖 Natural Language to SQL**: Convert plain English questions into optimized SQL queries
- **📊 Intelligent Visualizations**: Automatic chart generation based on data types and context
- **🔒 Enterprise Security**: Role-based access control, data masking, and audit logging
- **💬 Conversational Interface**: Streamlit-based chat UI for seamless interaction
- **🚀 High Performance**: Async FastAPI backend with vector search capabilities
- **🔍 Semantic Search**: Milvus-powered vector database for contextual query understanding
- **⚡ Background Processing**: Celery integration for long-running analytics tasks

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   FastAPI API   │    │     MySQL DB    │
│   (Frontend)    │◄──►│   (Backend)     │◄──►│   (Data Store)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │   LangChain     │              │
         └──────────────┤   AI Agents     │◄─────────────┘
                        └─────────────────┘
                                 │
                   ┌─────────────────┐    ┌─────────────────┐
                   │   Milvus VDB    │    │   OpenAI API    │
                   │  (Embeddings)   │    │     (LLM)       │
                   └─────────────────┘    └─────────────────┘
```

### 🧩 Core Components

- **AI Module**: LangChain-based agents for SQL generation and chart recommendations
- **Security Module**: SQL injection protection, permission management, and data masking
- **API Layer**: RESTful endpoints for chat, data queries, and user management
- **UI Layer**: Interactive Streamlit interface with real-time chat capabilities
- **Services**: Business logic for chat, data processing, and caching

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **MySQL 8.0+**
- **Milvus 2.3+** (for vector search)
- **Redis** (optional, for caching)
- **OpenAI API Key**

### 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/easy-chatbi.git
   cd easy-chatbi
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment**
   ```bash
   python scripts/setup.py
   ```

5. **Populate sample data**
   ```bash
   python scripts/seed_data.py
   ```

6. **Start the platform**
   ```bash
   python scripts/run.py dev
   ```

Visit:
- **UI**: http://localhost:8501
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Database Configuration
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=chatbi

# Vector Database
MILVUS_HOST=localhost
MILVUS_PORT=19530

# AI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4

# Security
SECRET_KEY=your_secret_key_here

# Application
DEBUG=True
LOG_LEVEL=INFO
API_PORT=8000
UI_PORT=8501

# Optional: Redis
REDIS_URL=redis://localhost:6379
```

### Permission Configuration

Edit `data/permissions.yaml` to customize:
- User roles and permissions
- Database access rules
- SQL security policies
- Data masking rules

## 👥 Default Users

| Username | Password   | Role      | Description |
|----------|------------|-----------|-------------|
| admin    | admin123   | admin     | Full system access |
| analyst  | analyst123 | analyst   | Data analysis permissions |
| viewer   | viewer123  | viewer    | Read-only access |
| demo     | demo123    | guest     | Limited demo access |

## 📖 Usage Guide

### 💬 Chat Interface

1. **Login** with any of the default credentials
2. **Ask questions** in natural language:
   - "Show me sales trends for the last 6 months"
   - "Which customers have the highest revenue?"
   - "Create a chart of products by category"

3. **Review generated SQL** and approve execution
4. **Interact with results** through charts and tables

### 🔧 API Usage

#### Authentication
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "analyst", "password": "analyst123"}'
```

#### Chat Completion
```bash
curl -X POST "http://localhost:8000/chat/complete" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "Show me total sales by month", "session_id": "uuid"}'
```

#### Data Query
```bash
curl -X POST "http://localhost:8000/data/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"sql": "SELECT COUNT(*) FROM customers"}'
```

## 🔒 Security Features

### Access Control
- **Role-based permissions** with granular database access
- **SQL injection protection** through query validation
- **Row-level security** for sensitive data
- **Rate limiting** to prevent abuse

### Data Protection
- **Field-level masking** for PII data
- **Audit logging** for compliance
- **IP restrictions** for admin access
- **Session management** with secure tokens

### SQL Security
- **Whitelist approach** for allowed SQL functions
- **Blocked operations** (DROP, DELETE, etc.)
- **Query timeout** limits
- **Result size** restrictions

## 🚀 Deployment

### Docker Deployment

1. **Build and start services**
   ```bash
   docker-compose up -d
   ```

2. **Initialize database**
   ```bash
   docker-compose exec app python scripts/setup.py
   docker-compose exec app python scripts/seed_data.py
   ```

### Production Considerations

- Use **PostgreSQL** or **MySQL** in production
- Configure **Redis** for session storage
- Set up **SSL/TLS** certificates
- Use **environment-specific** configuration files
- Implement **monitoring** and **logging**
- Configure **backup** strategies

## 🛠️ Development

### Project Structure

```
easy-chatbi/
├── api/                    # FastAPI backend
│   ├── main.py            # Application entry point
│   ├── database.py        # Database connections
│   ├── models.py          # SQLAlchemy models
│   └── routes/            # API endpoints
├── ai/                    # AI/ML components
│   ├── llm_client.py      # LLM integration
│   ├── sql_agent.py       # SQL generation
│   └── tools/             # AI tools
├── ui/                    # Streamlit frontend
│   ├── app.py             # Main UI application
│   ├── pages/             # UI pages
│   └── components/        # Reusable components
├── security/              # Security modules
├── services/              # Business logic
├── tasks/                 # Background tasks
└── utils/                 # Utilities
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/

# Run with coverage
pytest --cov=api --cov=ai tests/
```

### Adding New Features

1. **AI Agents**: Extend `ai/` modules for new analysis capabilities
2. **API Endpoints**: Add routes in `api/routes/`
3. **UI Components**: Create new Streamlit components in `ui/components/`
4. **Security Rules**: Update `data/permissions.yaml`

## 📊 Monitoring & Logging

### Application Logs
```bash
# View real-time logs
tail -f logs/chatbi.log

# Filter by level
grep "ERROR" logs/chatbi.log
```

### Health Checks
- **API Health**: `GET /health`
- **Database**: Connection status in logs
- **Vector DB**: Milvus collection status

### Metrics
- Query execution times
- User session analytics
- Error rates and types
- Resource utilization

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```bash
   # Check MySQL service
   sudo systemctl status mysql
   
   # Verify credentials
   mysql -u root -p -h localhost
   ```

2. **Milvus Connection Error**
   ```bash
   # Check Milvus status
   docker ps | grep milvus
   
   # Restart Milvus
   docker-compose restart milvus
   ```

3. **OpenAI API Issues**
   ```bash
   # Test API key
   curl -H "Authorization: Bearer $OPENAI_API_KEY" \
        https://api.openai.com/v1/models
   ```

4. **Streamlit Not Loading**
   ```bash
   # Check port availability
   netstat -tlnp | grep 8501
   
   # Clear Streamlit cache
   streamlit cache clear
   ```

### Debug Mode

```bash
# Run with debug logging
LOG_LEVEL=DEBUG python scripts/run.py dev

# Run individual services
python scripts/run.py api    # API only
python scripts/run.py ui     # UI only
```

## 🤝 Contributing

### Development Workflow

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Add** tests for new functionality
5. **Update** documentation
6. **Submit** a pull request

### Code Standards

- **Python**: Follow PEP 8
- **Type hints**: Required for new code
- **Documentation**: Docstrings for all functions
- **Testing**: Unit tests for core functionality

### Commit Convention

```
feat: add new chart type support
fix: resolve SQL injection vulnerability
docs: update API documentation
test: add integration tests for chat
```

## 📝 API Documentation

### Endpoints Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/login` | POST | User authentication |
| `/chat/sessions` | GET | List chat sessions |
| `/chat/complete` | POST | Process chat message |
| `/data/query` | POST | Execute SQL query |
| `/data/export` | POST | Export query results |
| `/admin/users` | GET | Manage users (admin only) |

### WebSocket Support

Real-time chat updates via WebSocket:
```javascript
const ws = new WebSocket('ws://localhost:8000/chat/ws');
ws.onmessage = (event) => {
    const response = JSON.parse(event.data);
    // Handle streaming response
};
```

## 🎯 Roadmap

### Version 2.0
- [ ] Multi-database support (PostgreSQL, BigQuery)
- [ ] Advanced visualization templates
- [ ] Custom dashboard builder
- [ ] Export to BI tools integration

### Version 2.5
- [ ] Machine learning model training interface
- [ ] Automated insight generation
- [ ] Advanced security features (RBAC, SAML)
- [ ] Multi-tenant architecture

### Version 3.0
- [ ] Real-time data streaming
- [ ] Mobile application
- [ ] Advanced natural language understanding
- [ ] Integration marketplace

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** for the AI framework
- **FastAPI** for the robust backend
- **Streamlit** for the intuitive UI
- **Milvus** for vector search capabilities
- **OpenAI** for language model integration

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-org/easy-chatbi/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/easy-chatbi/discussions)
- **Email**: support@chatbi.com
- **Documentation**: [Wiki](https://github.com/your-org/easy-chatbi/wiki)

---

**Built with ❤️ for democratizing data analysis through AI**