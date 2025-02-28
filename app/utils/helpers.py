from datetime import datetime
from typing import Dict, Any

def format_response(data: Any, message: str = "Success") -> Dict:
    return {
        "status": "success",
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "data": data
    } 