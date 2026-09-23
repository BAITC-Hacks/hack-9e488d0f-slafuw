FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN useradd --create-home app && mkdir -p /app/var && chown -R app:app /app
USER app
EXPOSE 8000
CMD ["python", "-m", "eventmatch", "serve", "--host", "0.0.0.0", "--port", "8000"]
