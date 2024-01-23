from fastapi import FastAPI
import psycopg2
from pydantic import BaseModel
import datetime

app = FastAPI()

#Валидация возвращаемых запросов по booking
#Наследование от BaseModel pydantic обозначает что выходные данные будут иметь определеный шаблон прописанный в классе
class BookingGet(BaseModel):
    id: int
    facility_id: int
    member_id: int 
    start_time: datetime.datetime
    slots: int
    
    #Настройки
    class Config:
        orm_mode = True
    

class SimpleUser(BaseModel):
    name : str
    surname : str


@app.get("/")
def say_hello():
    return "hello"

@app.get("/summ")
def summ_two(a: int, b: int) -> int:
    return a+b

#Фигурные скобки в url вытягивают значение в функцию
@app.get("/print/{number}")
def print_num(number: int):
    return number*2

#Name и дугие аргументы передаются в endpoint как параметр, тоесть .../user?name=Alexei
@app.post("/user")
def print_user(name: str):
    return {"message": f"Hello {name}"}

@app.get("/booking/all", response_model = BookingGet)
def all_bookings():
    conn = psycopg2.connect("postgresql://postgres:password@localhost:5432/exercises")        
    cursor = conn.cursor()
    cursor.execute("""
                   SELECT *
                   FROM cd.bookings
                   """)
    return cursor.fetchall()



#Валиация строковых значений
@app.post("/user/validate")
#Мы передали что User будет обязательно типа SimleUser(а именно тип нашаго описанного экземпляра класса где name:str and surname:str)
def user_validate(user: SimpleUser):
#Под капотом fastApi поймет что на вход мы получим json и провалидирует против класса SimpleUser
#А полученный результат будет экземпляром класса SimpleUser
    return "ok"