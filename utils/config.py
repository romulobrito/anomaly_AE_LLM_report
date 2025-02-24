import os
from dotenv import load_dotenv

def load_config():
    """
    Carrega variáveis de ambiente para OpenRouter/Deepseek
    """
    load_dotenv()
    
    # Verifica configurações necessárias
    api_key = os.getenv('OPENROUTER_API_KEY')
    base_url = os.getenv('BASE_URL')
    
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY não encontrada")
    
    return {
        'api_key': api_key,
        'base_url': base_url,
        'site_url': os.getenv('SITE_URL'),
        'site_name': os.getenv('SITE_NAME')
    }