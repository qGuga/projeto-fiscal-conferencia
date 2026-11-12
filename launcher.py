import logging
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

import streamlit.web.cli as stcli


def get_app_path():
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return root / "app.py"


def choose_port():
    for port in range(8501, 8511):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            try:
                server_socket.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("Não encontrei uma porta livre entre 8501 e 8510.")


def open_browser_when_ready(url):
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                webbrowser.open(url)
                return
        except Exception:
            time.sleep(0.5)
    logging.error("O servidor não respondeu no tempo esperado: %s", url)


def main():
    log_dir = Path(os.getenv("LOCALAPPDATA", Path.home())) / "ConferenciaFiscal"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "launcher.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    app_path = get_app_path()
    if not app_path.is_file():
        raise FileNotFoundError(f"Arquivo do aplicativo não encontrado: {app_path}")

    port = choose_port()
    url = f"http://127.0.0.1:{port}"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
    threading.Thread(target=open_browser_when_ready, args=(url,), daemon=True).start()

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--server.headless=true",
        "--global.developmentMode=false",
        "--browser.gatherUsageStats=false",
    ]
    logging.info("Iniciando Conferência Fiscal em %s", url)
    stcli.main()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Falha ao iniciar Conferência Fiscal")
        raise