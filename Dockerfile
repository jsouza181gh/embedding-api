# Servidor de embeddings (e5-large, CPU). Contexto de build: a raiz deste repositório.
#   docker build -t embedding-api .
#   docker run -p 8080:8080 -e EMBED_API_KEY=... embedding-api
# NÃO TESTADO: o build ainda não foi executado.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# modelo (~2,2 GB) baixado no build: o container sobe sem depender de rede/Hugging Face
ENV HF_HOME=/models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-large')"

COPY src src
RUN useradd --system --uid 10001 app && chown -R app /models
USER app

ENV HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4).status==200 else 1)"
# UM processo: cada worker carregaria uma cópia do modelo
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
