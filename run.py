import uvicorn
from multiprocessing import Process
from app.main import app as fastapi_app
from app.dashboard.server import init_dashboard
import webbrowser
import time
from threading import Timer
import platform
import os
import signal
import subprocess

def get_chrome_path():
    """Retorna o caminho do Chrome baseado no sistema operacional"""
    system = platform.system().lower()
    if system == 'linux':
        # Tenta diferentes localizações comuns do Chrome no Linux
        chrome_paths = [
            'google-chrome',
            'google-chrome-stable',
            '/usr/bin/google-chrome',
            '/usr/bin/google-chrome-stable'
        ]
        for path in chrome_paths:
            try:
                subprocess.run([path, '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return f"{path} %s"
            except:
                continue
        return None
    elif system == 'windows':
        return 'C:/Program Files/Google/Chrome/Application/chrome.exe %s'
    elif system == 'darwin':  # MacOS
        return 'open -a /Applications/Google\ Chrome.app %s'
    return None

def open_browser():
    """Abre os URLs no Chrome"""
    urls = [
        "http://localhost:8000/docs",
        "http://localhost:8050"
    ]
    
    try:
        # Tenta usar Chrome
        chrome_path = get_chrome_path()
        if chrome_path:
            # Configura o Chrome como navegador padrão para esta sessão
            browser = webbrowser.get(chrome_path)
            
            # Abre cada URL em uma nova aba do Chrome
            first_url = True
            for url in urls:
                if first_url:
                    browser.open_new(url)  # Abre primeira URL em nova janela
                    first_url = False
                    time.sleep(1)  # Pequena pausa para garantir que o Chrome abriu
                else:
                    browser.open_new_tab(url)  # Abre demais URLs em novas abas
                time.sleep(0.5)  # Pequena pausa entre aberturas
        else:
            print("Chrome não encontrado. Tentando navegador padrão...")
            for url in urls:
                webbrowser.open(url)
    except Exception as e:
        print(f"Erro ao abrir navegador: {e}")
        print("Por favor, abra manualmente os URLs:")
        for url in urls:
            print(f"  - {url}")

def run_fastapi():
    """Executa o servidor FastAPI"""
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",  # Mudado de 0.0.0.0 para 127.0.0.1
        port=8000, 
        reload=True,
        access_log=True  # Adiciona logs para debug
    )

def run_dashboard():
    """Executa o servidor Dash"""
    try:
        dash_app = init_dashboard()
        dash_app.run_server(
            host="127.0.0.1",
            port=8050, 
            debug=True,
            use_reloader=False  # Desabilita o reloader para evitar processos duplicados
        )
    except Exception as e:
        print(f"Erro ao iniciar Dashboard: {e}")

def check_port(port):
    """Verifica se a porta está em uso"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))  # Mudado para 127.0.0.1
    sock.close()
    return result == 0

def wait_for_server(port, timeout=30):
    """Espera até que o servidor esteja respondendo"""
    import requests
    from requests.exceptions import RequestException
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            if port == 8000:
                requests.get(f"http://127.0.0.1:{port}/docs")
            else:
                requests.get(f"http://127.0.0.1:{port}")
            return True
        except RequestException:
            time.sleep(1)
            print(f"Aguardando servidor na porta {port}...")
    return False

def kill_process_on_port(port):
    """Mata processos que estejam usando uma porta específica"""
    try:
        # Encontra o PID do processo usando a porta
        cmd = f"sudo lsof -t -i:{port}"  # Adicionado sudo
        pids = subprocess.check_output(cmd, shell=True).decode().split()
        
        # Mata cada processo com SIGKILL
        for pid in pids:
            try:
                # Usa sudo para garantir permissões
                subprocess.run(['sudo', 'kill', '-9', pid], check=True)
                print(f"Processo {pid} na porta {port} foi encerrado")
            except subprocess.CalledProcessError:
                print(f"Erro ao matar processo {pid}")
            except Exception as e:
                print(f"Erro inesperado ao matar processo {pid}: {e}")
    except subprocess.CalledProcessError:
        # Nenhum processo encontrado na porta
        pass
    except Exception as e:
        print(f"Erro ao verificar porta {port}: {e}")

def clean_database():
    """Limpa o banco de dados SQLite"""
    try:
        if os.path.exists("sql_app.db"):
            os.remove("sql_app.db")
            print("Banco de dados limpo!")
    except Exception as e:
        print(f"Erro ao limpar banco de dados: {e}")

def kill_all_processes():
    """Mata todos os processos nas portas 8000 e 8050"""
    try:
        # Primeiro mata os processos do dashboard
        kill_process_on_port(8050)
        time.sleep(1)  # Pequena pausa para garantir que os processos foram encerrados
        
        # Depois mata os processos da API
        kill_process_on_port(8000)
        time.sleep(1)
        
    except Exception as e:
        print(f"Erro ao matar processos: {e}")

def clean_and_init():
    """Limpa o banco de dados e inicializa a aplicação"""
    try:
        # Primeiro mata todos os processos
        kill_all_processes()
        
        # Depois limpa o banco de dados
        clean_database()
        
        # Verifica se as portas estão realmente livres
        if check_port(8000) or check_port(8050):
            print("ERRO: Não foi possível liberar as portas!")
            return False
            
        return True
        
    except Exception as e:
        print(f"Erro na inicialização: {e}")
        return False

if __name__ == "__main__":
    # Inicialização
    if not clean_and_init():
        exit(1)

    try:
        # Iniciar FastAPI primeiro
        print("Iniciando servidor FastAPI em http://127.0.0.1:8000")
        fastapi_process = Process(target=run_fastapi)
        fastapi_process.start()
        
        # Esperar FastAPI iniciar
        if not wait_for_server(8000):
            print("Erro: FastAPI não iniciou corretamente!")
            fastapi_process.terminate()
            exit(1)
        print("FastAPI iniciado com sucesso!")
        
        # Depois iniciar o Dashboard
        print("Iniciando Dashboard em http://127.0.0.1:8050")
        dashboard_process = Process(target=run_dashboard)
        dashboard_process.start()
        
        # Esperar Dashboard iniciar
        if not wait_for_server(8050):
            print("Erro: Dashboard não iniciou corretamente!")
            fastapi_process.terminate()
            dashboard_process.terminate()
            exit(1)
        print("Dashboard iniciado com sucesso!")

        # Abrir navegador por último
        print("Abrindo navegador...")
        Timer(2, open_browser).start()

        # Aguardar os processos indefinidamente
        while True:
            if not fastapi_process.is_alive() or not dashboard_process.is_alive():
                print("Um dos servidores parou inesperadamente!")
                break
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nEncerrando servidores...")
    except Exception as e:
        print(f"\nErro: {e}")
    finally:
        # Garantir que os processos sejam encerrados
        try:
            fastapi_process.terminate()
            dashboard_process.terminate()
            fastapi_process.join(timeout=5)
            dashboard_process.join(timeout=5)
            print("Servidores encerrados!")
        except:
            print("Erro ao encerrar servidores!") 