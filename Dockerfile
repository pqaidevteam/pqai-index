FROM python:3.7
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
COPY ./ /app/
#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8258"]
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--reload"] # For dev
