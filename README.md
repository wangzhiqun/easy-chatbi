# ChatBI - Data Intelligence Platform 

ChatBI is an AI-powered data intelligence platform that enables natural language interaction with your databases. Built with modern Python technologies including LangChain, FastAPI, Streamlit, and the Model Context Protocol (MCP).

## ✨ Features

- **Natural Language Queries**: Ask questions about your data in plain language
- **AI-Powered SQL Generation**: Automatically generate SQL queries from natural language
- **Interactive Visualizations**: Create charts and dashboards with AI recommendations
- **Data Analysis**: Comprehensive analysis including correlations, trends, and anomalies
- **MCP Integration**: Model Context Protocol support for advanced tool usage
- **Multi-Database Support**: Connect to MySQL, PostgreSQL, and more
- **Real-time Chat Interface**: Interactive conversation with context retention
- **Security**: SQL injection protection and query validation

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional, for databases)
- OpenAI API Key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/easy-chatbi.git
cd easy-chatbi
```

2. **Run the setup script**
```bash
python scripts/setup.py
```

3. **Configure environment variables**

Edit `.env` file with your settings:
```env
OPENAI_API_KEY=your_openai_api_key_here
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=chatbi
MYSQL_USER=root
MYSQL_PASSWORD=password
```

4. **Seed the database (optional)**
```bash
python scripts/seed_data.py
```

5. **Start the application**
```bash
python scripts/run.py
```

This starts:
- API server at http://localhost:8000
- Streamlit UI at http://localhost:8501
- API documentation at http://localhost:8000/docs

## 🛠️ Development

### Building Docker Image

```shell
docker build -t chatbi .
docker run -p 8000:8000 -p 8501:8501 chatbi
```

```shell
docker-compose up -d
```

## 🔧 Configuration

### Database Configuration

ChatBI supports multiple databases. Configure in `.env`:

```env
# MySQL
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=chatbi
MYSQL_USER=root
MYSQL_PASSWORD=password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

### AI Configuration

```env
# OpenAI
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini

# Generation settings
TEMPERATURE=0.7
MAX_TOKENS=2000
```

## 🔒 Security

- SQL injection protection
- Query validation and sanitization
- API key authentication
- Rate limiting
- Secure password hashing

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**ChatBI** - Making data conversations intelligent 🚀