import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class ChatBIRunner:

    def __init__(self):
        self.processes = []
        self.running = True

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        print("\n\n🛑 Shutting down ChatBI services...")
        self.running = False
        self.stop_all()
        sys.exit(0)

    def start_api(self, host="0.0.0.0", port=8000):
        print(f"\n🚀 Starting API server on {host}:{port}...")

        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "api.main:app",
            "--host", host,
            "--port", str(port),
            "--reload"
        ]

        process = subprocess.Popen(cmd)
        self.processes.append(process)

        print(f"✅ API server started (PID: {process.pid})")
        print(f"📍 API Docs: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs")

        return process

    def start_streamlit(self, port=8501):
        print(f"\n🎨 Starting Streamlit UI on port {port}...")

        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "ui/app.py",
            "--server.port", str(port),
        ]

        process = subprocess.Popen(cmd)
        self.processes.append(process)

        print(f"✅ Streamlit UI started (PID: {process.pid})")
        print(f"📍 UI URL: http://localhost:{port}")

        return process

    def start_mcp_server(self):
        print("\n🔌 Starting MCP server...")

        cmd = [
            sys.executable,
            "-m",
            "mcpv1.server"
        ]

        process = subprocess.Popen(cmd)
        self.processes.append(process)

        print(f"✅ MCP server started (PID: {process.pid})")

        return process

    def monitor_processes(self):
        print("\n📊 Monitoring services...")
        print("Press Ctrl+C to stop all services\n")

        while self.running:
            for process in self.processes:
                if process.poll() is not None:
                    print(f"⚠️  Process {process.pid} has stopped")
                    self.processes.remove(process)

            time.sleep(1)

    def stop_all(self):
        for process in self.processes:
            if process.poll() is None:
                print(f"  Stopping process {process.pid}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

        self.processes.clear()

    def run(self, services=['api', 'ui'], auto_open=True):
        print("""
╔═══════════════════════════════════════╗
║       ChatBI Platform Runner          ║
╚═══════════════════════════════════════╝
        """)

        try:
            if 'api' in services:
                self.start_api()
                time.sleep(2)

            if 'ui' in services:
                self.start_streamlit()
                time.sleep(3)

            if 'mcp' in services:
                self.start_mcp_server()

            print("""
╔═══════════════════════════════════════╗
║     All services started! 🎉          ║
╚═══════════════════════════════════════╝

Services running:
  📡 API: http://localhost:8000
  📊 API Docs: http://localhost:8000/docs
  🎨 UI: http://localhost:8501

Press Ctrl+C to stop all services
            """)

            self.monitor_processes()

        except KeyboardInterrupt:
            pass
        finally:
            self.stop_all()


def main():
    parser = argparse.ArgumentParser(description="Run ChatBI services")
    parser.add_argument(
        "--services",
        nargs='+',
        choices=['api', 'ui', 'mcp', 'all'],
        default=['api', 'ui'],
        help="Services to start"
    )
    parser.add_argument("--api-host", default="0.0.0.0", help="API host")
    parser.add_argument("--api-port", type=int, default=8000, help="API port")
    parser.add_argument("--ui-port", type=int, default=8501, help="UI port")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    parser.add_argument("--dev", action="store_true", help="Run in development mode")

    args = parser.parse_args()

    services = args.services
    if 'all' in services:
        services = ['api', 'ui', 'mcp']

    runner = ChatBIRunner()

    if args.dev:
        os.environ['ENVIRONMENT'] = 'development'

    runner.run(
        services=services,
        auto_open=not args.no_browser
    )


if __name__ == "__main__":
    main()
