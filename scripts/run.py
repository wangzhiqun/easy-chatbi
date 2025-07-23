#!/usr/bin/env python3
"""
Startup script for ChatBI platform
Manages application lifecycle and service orchestration
"""

import os
import sys
import signal
import subprocess
import time
import argparse
import threading
from pathlib import Path
from typing import List, Dict, Optional
import psutil
import requests

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.config import Config
from utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)


class ServiceManager:
    """Manage application services and processes"""

    def __init__(self, config: Config):
        self.config = config
        self.processes: Dict[str, subprocess.Popen] = {}
        self.shutdown_event = threading.Event()
        self.project_root = project_root

    def check_environment(self) -> bool:
        """Check if environment is properly set up"""
        logger.info("Checking environment setup...")

        # Check Python version
        if sys.version_info < (3, 11):
            logger.error("Python 3.11+ is required")
            return False

        # Check required files
        required_files = [
            '.env',
            'requirements.txt'
        ]

        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                logger.error(f"Required file missing: {file_path}")
                return False

        # Check database connection
        try:
            import pymysql
            connection = pymysql.connect(
                host=self.config.DB_HOST,
                port=self.config.DB_PORT,
                user=self.config.DB_USER,
                password=self.config.DB_PASSWORD,
                database=self.config.DB_NAME
            )
            connection.close()
            logger.info("Database connection: OK")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

        # Check Milvus connection
        try:
            from pymilvus import connections
            connections.connect(
                alias="test",
                host=self.config.MILVUS_HOST,
                port=self.config.MILVUS_PORT
            )
            connections.disconnect("test")
            logger.info("Milvus connection: OK")
        except Exception as e:
            logger.error(f"Milvus connection failed: {e}")
            return False

        # Check OpenAI API key
        if not self.config.OPENAI_API_KEY or self.config.OPENAI_API_KEY == "your_openai_api_key":
            logger.error("OpenAI API key not configured")
            return False

        logger.info("Environment check completed successfully")
        return True

    def start_api_server(self) -> bool:
        """Start FastAPI server"""
        try:
            cmd = [
                sys.executable, "-m", "uvicorn",
                "api.main:app",
                "--host", "0.0.0.0",
                "--port", str(self.config.API_PORT),
                "--reload" if self.config.DEBUG else "--no-reload"
            ]

            if self.config.DEBUG:
                cmd.extend(["--log-level", "debug"])

            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )

            self.processes['api'] = process
            logger.info(f"API server started on port {self.config.API_PORT}")

            # Wait a moment and check if process is still running
            time.sleep(2)
            if process.poll() is not None:
                logger.error("API server failed to start")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            return False

    def start_streamlit_ui(self) -> bool:
        """Start Streamlit UI"""
        try:
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                "ui/app.py",
                "--server.port", str(self.config.UI_PORT),
                "--server.address", "0.0.0.0",
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false"
            ]

            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )

            self.processes['ui'] = process
            logger.info(f"Streamlit UI started on port {self.config.UI_PORT}")

            # Wait a moment and check if process is still running
            time.sleep(3)
            if process.poll() is not None:
                logger.error("Streamlit UI failed to start")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to start Streamlit UI: {e}")
            return False

    def start_celery_worker(self) -> bool:
        """Start Celery worker for background tasks"""
        try:
            cmd = [
                sys.executable, "-m", "celery",
                "worker",
                "-A", "tasks.celery_app",
                "--loglevel=info"
            ]

            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )

            self.processes['celery'] = process
            logger.info("Celery worker started")

            # Wait a moment and check if process is still running
            time.sleep(2)
            if process.poll() is not None:
                logger.error("Celery worker failed to start")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to start Celery worker: {e}")
            return False

    def wait_for_service(self, url: str, timeout: int = 30) -> bool:
        """Wait for a service to become available"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass

            time.sleep(1)

        return False

    def monitor_processes(self):
        """Monitor running processes and restart if needed"""
        while not self.shutdown_event.is_set():
            for service_name, process in list(self.processes.items()):
                if process.poll() is not None:
                    logger.warning(f"Service {service_name} has stopped unexpectedly")
                    # In production, you might want to restart the service
                    del self.processes[service_name]

            time.sleep(5)

    def shutdown_services(self):
        """Gracefully shutdown all services"""
        logger.info("Shutting down services...")
        self.shutdown_event.set()

        for service_name, process in self.processes.items():
            logger.info(f"Stopping {service_name}...")

            try:
                # Try graceful shutdown first
                process.terminate()

                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    logger.warning(f"Force killing {service_name}")
                    process.kill()
                    process.wait()

                logger.info(f"Service {service_name} stopped")

            except Exception as e:
                logger.error(f"Error stopping {service_name}: {e}")

        self.processes.clear()
        logger.info("All services stopped")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_services()
        sys.exit(0)

    def print_status(self):
        """Print application status and URLs"""
        logger.info("\n" + "=" * 50)
        logger.info("🚀 ChatBI Platform is running!")
        logger.info("=" * 50)

        if 'api' in self.processes:
            logger.info(f"📡 API Server: http://localhost:{self.config.API_PORT}")
            logger.info(f"📋 API Docs: http://localhost:{self.config.API_PORT}/docs")

        if 'ui' in self.processes:
            logger.info(f"🎨 Streamlit UI: http://localhost:{self.config.UI_PORT}")

        if 'celery' in self.processes:
            logger.info("⚙️  Celery Worker: Running")

        logger.info("\n💡 Default login credentials:")
        logger.info("   - admin/admin123 (Administrator)")
        logger.info("   - analyst/analyst123 (Data Analyst)")
        logger.info("   - viewer/viewer123 (Viewer)")
        logger.info("   - demo/demo123 (Guest)")

        logger.info("\n🛑 Press Ctrl+C to stop all services")
        logger.info("=" * 50)


def run_development():
    """Run in development mode with all services"""
    config = Config()
    manager = ServiceManager(config)

    # Setup signal handlers
    signal.signal(signal.SIGINT, manager.signal_handler)
    signal.signal(signal.SIGTERM, manager.signal_handler)

    try:
        # Check environment
        if not manager.check_environment():
            logger.error("Environment check failed. Please run setup.py first.")
            return False

        # Start services
        logger.info("Starting ChatBI platform in development mode...")

        success = True

        # Start API server
        if not manager.start_api_server():
            success = False

        # Start Streamlit UI
        if success and not manager.start_streamlit_ui():
            success = False

        # Start Celery worker (optional)
        if success:
            try:
                manager.start_celery_worker()
            except Exception as e:
                logger.warning(f"Celery worker failed to start: {e}")
                logger.info("Continuing without background task processing")

        if not success:
            manager.shutdown_services()
            return False

        # Wait for services to be ready
        logger.info("Waiting for services to be ready...")

        api_ready = manager.wait_for_service(f"http://localhost:{config.API_PORT}/health")
        ui_ready = manager.wait_for_service(f"http://localhost:{config.UI_PORT}")

        if not api_ready:
            logger.error("API server failed to become ready")
            manager.shutdown_services()
            return False

        if not ui_ready:
            logger.warning("Streamlit UI may not be ready yet")

        # Print status
        manager.print_status()

        # Start monitoring thread
        monitor_thread = threading.Thread(target=manager.monitor_processes, daemon=True)
        monitor_thread.start()

        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

        return True

    except Exception as e:
        logger.error(f"Failed to start platform: {e}")
        manager.shutdown_services()
        return False
    finally:
        manager.shutdown_services()


def run_api_only():
    """Run only the API server"""
    config = Config()
    manager = ServiceManager(config)

    # Setup signal handlers
    signal.signal(signal.SIGINT, manager.signal_handler)
    signal.signal(signal.SIGTERM, manager.signal_handler)

    try:
        if not manager.check_environment():
            return False

        logger.info("Starting API server only...")

        if not manager.start_api_server():
            return False

        if not manager.wait_for_service(f"http://localhost:{config.API_PORT}/health"):
            logger.error("API server failed to become ready")
            return False

        logger.info(f"🚀 API Server is running on http://localhost:{config.API_PORT}")
        logger.info(f"📋 API Documentation: http://localhost:{config.API_PORT}/docs")
        logger.info("🛑 Press Ctrl+C to stop")

        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

        return True

    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        return False
    finally:
        manager.shutdown_services()


def run_ui_only():
    """Run only the Streamlit UI"""
    config = Config()
    manager = ServiceManager(config)

    # Setup signal handlers
    signal.signal(signal.SIGINT, manager.signal_handler)
    signal.signal(signal.SIGTERM, manager.signal_handler)

    try:
        logger.info("Starting Streamlit UI only...")

        if not manager.start_streamlit_ui():
            return False

        if not manager.wait_for_service(f"http://localhost:{config.UI_PORT}"):
            logger.warning("Streamlit UI may not be ready yet")

        logger.info(f"🚀 Streamlit UI is running on http://localhost:{config.UI_PORT}")
        logger.info("🛑 Press Ctrl+C to stop")

        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

        return True

    except Exception as e:
        logger.error(f"Failed to start Streamlit UI: {e}")
        return False
    finally:
        manager.shutdown_services()


def check_setup():
    """Check if the platform is properly set up"""
    config = Config()
    manager = ServiceManager(config)

    logger.info("Checking ChatBI platform setup...")

    if manager.check_environment():
        logger.info("✅ Platform is properly set up and ready to run")
        return True
    else:
        logger.error("❌ Platform setup is incomplete")
        logger.info("Please run the following commands:")
        logger.info("1. python scripts/setup.py")
        logger.info("2. python scripts/seed_data.py")
        return False


def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(description="ChatBI Platform Runner")
    parser.add_argument(
        "mode",
        choices=["dev", "api", "ui", "check"],
        default="dev",
        nargs="?",
        help="Run mode: dev (all services), api (API only), ui (UI only), check (verify setup)"
    )

    args = parser.parse_args()

    try:
        if args.mode == "dev":
            success = run_development()
        elif args.mode == "api":
            success = run_api_only()
        elif args.mode == "ui":
            success = run_ui_only()
        elif args.mode == "check":
            success = check_setup()
        else:
            parser.print_help()
            success = False

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()