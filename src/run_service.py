import uvicorn
from dotenv import load_dotenv

from core import settings

load_dotenv()

if __name__ == "__main__":
    print("RUNNING SERVICE")
    uvicorn.run("service:app", host=settings.HOST, port=settings.PORT, reload=settings.is_dev())
