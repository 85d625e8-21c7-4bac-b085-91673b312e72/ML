# Используем официальный образ Python
FROM python:3.10-slim

# Устанавливаем зависимости для cairo
RUN apt-get update && \
    apt-get install -y \
    libcairo2-dev \
    libpango1.0-dev \
    libssl-dev \
    libgirepository1.0-dev \
    pkg-config \
    python3-dev \
    gcc \
    && apt-get clean

# Устанавливаем Python пакеты
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Копируем исходный код в контейнер
COPY . /app

# Устанавливаем рабочую директорию
WORKDIR /app

# Команда для выполнения скрипта
CMD ["python", "generate.py"]
