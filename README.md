# embedding-api

API de embeddings (FastAPI, assíncrona) com `intfloat/multilingual-e5-large` (1024 dimensões, CPU). Serve a base de conhecimento de compliance do projeto DRX (`tools/compliance_kb`), mas não depende dele: recebe texto, devolve vetor.

Existe porque carregar o e5-large leva ~45 s por processo. Com o modelo residente, uma busca custa ~0,4 s de embedding.

## Contrato

```
POST /embed
Authorization: Bearer <EMBED_API_KEY>
{"texts": ["..."], "kind": "query" | "passage"}      # kind padrão: "query"; até 64 textos, até 8000 caracteres cada

200 {"model": "intfloat/multilingual-e5-large", "dim": 1024, "embeddings": [[...], ...]}
401 chave ausente/errada   422 validação (vazio, grande demais, kind inválido)   503 fila cheia (header Retry-After)

GET /health   (sem autenticação)
200 {"status": "ok", "model": "...", "dim": 1024, "pending": 0}
```

- O servidor aplica os prefixos do e5 (`query: ` / `passage: `); o cliente manda o texto puro.
- Vetores normalizados (cosseno == produto interno).
- `kind: "passage"` para indexar documentos, `"query"` para buscar. Misturar os dois piora a busca.
- O conteúdo dos textos não é logado.

## Concorrência

O event loop nunca executa o modelo: a inferência vai para um `ThreadPoolExecutor` (o torch libera o GIL). Pedidos simultâneos entram numa fila. `EMBED_WORKERS=1` por padrão, porque uma única inferência já usa todos os núcleos; mais workers só competem pela mesma CPU. Quando `EMBED_MAX_PENDING` é atingido, responde `503` com `Retry-After` em vez de crescer sem limite.

Rode **um** processo uvicorn (sem `--workers N`): cada worker carregaria outra cópia do modelo (~2,2 GB de RAM).

## Rodar local

```
python -m venv .venv && .venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
EMBED_API_KEY=dev-key uvicorn src.app:app --host 127.0.0.1 --port 8080
```

Teste rápido:

```
curl -s http://127.0.0.1:8080/embed -H "Authorization: Bearer dev-key" -H "Content-Type: application/json" \
     -d '{"texts":["consentimento do titular"],"kind":"query"}' | head -c 200
```

A primeira subida baixa o modelo (~2,2 GB) para o cache do Hugging Face. Atrás de proxy com certificado próprio o download pode falhar (`CERTIFICATE_VERIFY_FAILED`); depois de baixado, `HF_HUB_OFFLINE=1`.

## Testes

```
pytest
```

Usam um codificador falso: cobrem contrato, autenticação, validação, fila cheia e `/health` durante uma inferência lenta. **Não carregam o modelo.** A inferência com o modelo real foi verificada à mão (ver Medições).

## Deploy

`Dockerfile` na raiz (modelo baixado no build; o container não depende do Hugging Face para subir):

```
docker build -t embedding-api .
docker run -p 8080:8080 -e EMBED_API_KEY=... embedding-api
```

Com Docker Compose (`docker-compose.yml`: publica a porta **80** do host, `restart: unless-stopped`, limite de 6 GB):

```
cp .env.example .env && chmod 600 .env     # edite EMBED_API_KEY
docker compose up -d --build
curl -s http://localhost/health
```

Atualizar: `git pull && docker compose up -d --build`.

**O build do Dockerfile ainda não foi executado.** Para produção:
- HTTPS na frente (proxy reverso); o servidor fala HTTP puro. **Sem TLS, a chave Bearer e as consultas trafegam em claro**; o compose publica a porta 80 em HTTP, então só exponha assim numa rede de confiança ou adicione TLS.
- Chave longa e aleatória em `EMBED_API_KEY` (o servidor não sobe sem ela; `EMBED_ALLOW_NO_AUTH=1` existe só para dev local).
- ~3 GB de RAM livres e CPU dedicada. Sem GPU a latência depende dos núcleos.
- As consultas enviadas são derivadas do código revisado: restrinja a rede de acesso.
- Ajuste `EMBED_TORCH_THREADS` abaixo do número de núcleos se o `/health` ficar lento sob carga.

Variáveis: ver `.env.example`.

## Medições (desktop, CPU, cliente na mesma máquina)

| | tempo |
|---|---|
| servidor pronto (modelo em cache) | ~11 s |
| pedido sequencial | 0,38 a 0,56 s |
| 4 pedidos simultâneos | 0,91 s no total |
| 8 pedidos simultâneos | 1,55 s no total (mediana 1,18 s por pedido) |
| `/health` com 8 pedidos em andamento | respondeu em ~0,5 s (a CPU estava saturada) |

## Cliente (projeto DRX)

No `tools/compliance_kb/.env` do DRX:

```
KB_EMBED_URL=https://<host>
KB_EMBED_API_KEY=<a mesma chave>
```

O cliente (`embedder.py`) recusa respostas cujo `model` ou `dim` difira de `intfloat/multilingual-e5-large`/1024, porque vetores de outro modelo não são comparáveis com os do banco. Se mudar `EMBED_MODEL` aqui, é preciso migrar a coluna `vector(N)` e reingerir a base no DRX.
