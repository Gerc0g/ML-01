import requests
from loguru import logger

r = requests.get("http://localhost:8000/print/1/2")
#Пример получения статус кода
logger.info(r.status_code)
#Пример получения текста
logger.info(r.text)


#Пример передачи json
l = requests.post(
    "http://localhost:8000/user/validate",
    json = {'name' : 'Egor', 'surname': 'Logutov'}
    )

logger.info(f"Second status code: {l.status_code} /n Second text: {l.json()}")

