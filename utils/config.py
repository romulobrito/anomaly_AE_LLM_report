import os
from dotenv import load_dotenv

def load_config():
    """
    Carrega variáveis de ambiente
    """
    load_dotenv()
    
    # Verifica se a key existe
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY não encontrada nas variáveis de ambiente")
    
    return {
        'openai_api_key': api_key
    }