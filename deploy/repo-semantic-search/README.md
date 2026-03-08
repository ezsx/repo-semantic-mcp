# Repo Semantic Search Deploy Runbook

## Цель

Поднять standalone semantic MCP с отдельным `Qdrant`, чтобы агенты могли делать:
- semantic search по коду;
- semantic search по документации;
- hybrid search по смешанным RU/EN запросам.

Сервис специально разделяет две logical collections:
- `code`
- `docs`

Это сделано намеренно: агент должен явно понимать, что он ищет по коду или по документации,
а не всегда бить в один смешанный корпус.

## Что входит в `docs` collection

По умолчанию в `docs` коллекцию попадают:
- `docs/**`
- `agent_context/**`
- markdown/text файлы из других частей репозитория
- небольшие `.md` отчёты, включая `tools/testing/load/reports/**/*.md`

## Что входит в `code` collection

По умолчанию в `code` коллекцию попадают:
- `apps/**`
- `services/**`
- `libs/**`
- `deploy/**`
- `scripts/**`
- `tools/**` кроме файлов, которые классифицируются как docs/text

## Быстрый запуск shared HTTP режима

1. Скопировать `.env.example` в `.env`.
2. Запустить стек:

```bash
cd deploy/repo-semantic-search
docker compose -f docker-compose.repo-semantic-search.yml up -d --build
```

3. Дождаться первого auto-index.
4. Проверить HTTP endpoint:

```bash
curl http://localhost:8011/mcp
```

Либо использовать helper script, который ждёт именно рабочий MCP protocol:

```powershell
pwsh -File scripts/agents/ensure_repo_semantic_search.ps1 -Build
```

## GPU profile

Для машины с NVIDIA GPU используйте отдельный override и GPU env:

```powershell
Copy-Item deploy/repo-semantic-search/.env.gpu.example deploy/repo-semantic-search/.env.gpu
pwsh -File scripts/agents/ensure_repo_semantic_search.ps1 -Build -Gpu
```

Эквивалентный ручной запуск:

```bash
cd deploy/repo-semantic-search
docker compose -f docker-compose.repo-semantic-search.yml -f docker-compose.repo-semantic-search.gpu.yml up -d --build
```

Профиль рассчитан на `NVIDIA Container Toolkit` и TEI CUDA image. По официальному quick start TEI для GPU запускается через `--gpus all` и CUDA image семейства `cuda-1.9`. Источник: https://huggingface.co/docs/text-embeddings-inference/en/quick_tour

Логика профилей намеренно такая:
- CPU путь остаётся дефолтным и совместимым для коллег;
- GPU путь opt-in и включается только через `-Gpu`;
- helper script сначала ищет `deploy/repo-semantic-search/.env.gpu`, затем обычный `.env`.

### Почему по умолчанию используется TEI, а не FastEmbed

В спецификации целевая dense model — `bge-m3`, но на практике `fastembed_local`
не поддерживает её напрямую в текущем runtime, а `gte-multilingual-base` на CPU
даёт слишком долгий cold-start для первой полной индексации.

Поэтому рабочий default deployment идёт через отдельный `TEI` контейнер и
multilingual модель `intfloat/multilingual-e5-small`.

Это даёт:
- устойчивый контейнерный запуск;
- хороший mixed-language retrieval;
- приемлемое время первой индексации на CPU;
- сохранение архитектуры `Qdrant + separate embedding backend`.

`fastembed_local` остаётся как fallback/dev-режим, но не является основным
production-like профилем.

GPU profile сознательно вынесен отдельно:
- CPU default остаётся дешёвым и стабильным после ребута;
- GPU override позволяет поднять целевую multilingual модель `BAAI/bge-m3` без смены MCP contract.

## Локальный stdio режим

Для локального запуска без shared HTTP:

```bash
$env:SEMANTIC_MCP_TRANSPORT="stdio"
py -3.12 apps/repo-semantic-mcp/main.py
```

## Operational notes

- `Qdrant` должен жить на persistent volume.
- `TEI` также должен жить с persistent cache volume, иначе после очистки docker volume
  модель будет скачиваться заново.
- `SEMANTIC_MCP_AUTO_INDEX_ON_START=1` подходит для первого rollout и локальной машины.
- Для shared deployment при большом индексе можно выключить auto-index и запускать rebuild отдельно через tool.
- Если используется `tei_http`, сам TEI сервис нужно поднимать отдельным контейнером и указывать его URL через `SEMANTIC_MCP_TEI_URL`.
- В текущем compose TEI уже включён и стартует автоматически.

## Регистрация в Codex и Claude

После первого успешного запуска можно зарегистрировать shared HTTP MCP сразу в обеих
локальных конфигурациях:

```powershell
pwsh -File scripts/agents/register_repo_semantic_search.ps1
```

Скрипт обновляет:
- `%USERPROFILE%\.codex\config.toml`
- `%USERPROFILE%\.claude.json`

## Поведение после ребута машины

- Все контейнеры стека работают с `restart: unless-stopped`.
- Если Docker Desktop стартует вместе с Windows, semantic MCP поднимется автоматически.
- Если Docker Desktop был остановлен вручную или нужно перепроверить готовность,
  используйте:

```powershell
pwsh -File scripts/agents/ensure_repo_semantic_search.ps1
```

Для GPU-профиля:

```powershell
pwsh -File scripts/agents/ensure_repo_semantic_search.ps1 -Gpu
```
