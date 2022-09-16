FROM python:3.7
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
COPY ./core /app/
COPY ./main.py /app/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
