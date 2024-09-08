import os
from dotenv import load_dotenv
import requests
from tqdm import tqdm
from transformers import AutoTokenizer
from llama_cpp import Llama
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

load_dotenv()
app = FastAPI()

api_host = os.getenv("API_HOST")
api_port = os.getenv("API_PORT")
model_encoder = os.getenv("MODEL_ENCODER")
full_model = os.getenv("FULL_MODEL")
small_model = os.getenv("SMALL_MODEL")
tiny_model = os.getenv("TINY_MODEL")
current_model = os.getenv("CURRENT_MODEL")
hf_token = os.getenv("HF_TOKEN")

model_file = {
    "FULL_MODEL": full_model,
    "SMALL_MODEL": small_model,
    "TINY_MODEL": tiny_model
}.get(current_model, small_model)

if not os.path.exists(model_file):
    print(f"Файл модели {model_file} не найден, начинается загрузка...")
    
    def download_model(model_id, file, token):
        headers = {
            "Authorization": f"Bearer {token}"
        }
        
        url = f"https://huggingface.co/{model_id}/resolve/main/{file}"
        response = requests.get(url, headers=headers, stream=True)
        
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # Размер блока для загрузки

            with open(file, 'wb') as f, tqdm(
                total=total_size, unit='iB', unit_scale=True, desc=file
            ) as t:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        t.update(len(chunk))
                        
            print(f"Модель {file} успешно загружена.")
            return True
        else:
            print(f"Ошибка при загрузке модели: {response.status_code}")
            return False

    success = download_model(model_encoder, model_file, hf_token)

    if not success:
        print("Не удалось загрузить модель. Программа завершена.")
        exit(1)
else:
    print(f"Файл модели {model_file} уже существует.")

print(f"Модель {model_file} готова к использованию.")


tokenizer = AutoTokenizer.from_pretrained(
    model_encoder,
    cache_dir="./hf_cache",
    token=hf_token
)

print(f"Энкодер {model_file} готов к использованию.")

### MODEL

class PromptRequest(BaseModel):
    city: str

### END MODEL
### FUNCTIONS

def get_weather(city: str):
    """
    Функция, которая возвращает погоду в заданном городе.
    
    Args:
        city: Город, для которого надо узнать погоду.
    """
    import random
    
    return "sunny" if random.random() > 0.5 else "rainy"


def get_sunrise_sunset_times(city: str):
    """
    Функция, которая возвращает время восхода и заката для заданного города для текущей даты (дата от пользователя не требуется), в формате списка: [sunrise_time, sunset_time].
    
    Args:
        city: Город, в котором можно узнать время восхода и захода солнца.
    """

    return ["6:00", "18:00"]


### END FUNCTIONS

llm = Llama(
    model_path=model_file,
)

@app.post("/generate")
async def generate_response(request: Request):
    request_data = await request.json()
    user_query = request_data.get("query", "")

    messages = [
        {"role": "system", "content": "Ты - полезный помощник."},
        {"role": "user", "content": user_query}
    ]
    
    tools = [get_sunrise_sunset_times]
    
    PROMPT = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True
    )
    
    response = llm(
        PROMPT,
        max_tokens=256
    )

    return {"response": response}
