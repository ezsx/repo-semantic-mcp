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

1. Запустить helper script напрямую.
2. Явно указать target repo, который нужно индексировать.

```powershell
pwsh -File scripts/agents/ensure_repo_semantic_search.ps1 `
  -Build `
  -Profile cpu `
  -TargetRepoPath C:\nullexp_vpn\vpn-server
```

Для ручного compose-запуска:

```bash
cd deploy/repo-semantic-search
docker compose -f docker-compose.repo-semantic-search.yml --env-file .env.example up -d --build
```

После старта helper script ждёт готовый MCP protocol, а не только состояние контейнера.

Важно: один запущенный стек индексирует один target repo. Для переключения на другой
репозиторий нужно перезапустить тот же стек с новым `-TargetRepoPath`.

Коллекции теперь включают repo-specific key, поэтому `vpn-server` и любой другой
target repo не смогут тихо делить один и тот же индекс.

## GPU profile

Для машины с NVIDIA GPU используйте primary GPU profile:

```powershell
pwsh -File scripts/agents/ensure_repo_semantic_search.ps1 `
  -Build `
  -Profile gpu `
  -TargetRepoPath C:\nullexp_vpn\vpn-server
```

Этот профиль теперь основной:
- model: `Qwen/Qwen3-Embedding-0.6B`
- profile: `gpu_qwen3`

На текущей машине cold start этого профиля до готового MCP для `vpn-server`
составил примерно `8 минут`. Это нормальный startup budget, который нужно
закладывать в operational flow.

Эквивалентный ручной запуск:

```bash
cd deploy/repo-semantic-search
docker compose -f docker-compose.repo-semantic-search.yml -f docker-compose.repo-semantic-search.gpu.yml --env-file .env.gpu.example up -d --build
```

Профиль рассчитан на `NVIDIA Container Toolkit` и TEI CUDA image. По официальному quick start TEI для GPU запускается через `--gpus all` и CUDA image семейства `cuda-1.9`. Источник: https://huggingface.co/docs/text-embeddings-inference/en/quick_tour

Логика профилей намеренно такая:
- CPU путь остаётся дефолтным и совместимым для коллег;
- GPU путь opt-in и включается только через `-Profile gpu`;
- helper script умеет работать как с `.env*`, так и с `.env*.example`, поэтому первый запуск не требует ручного копирования env-файлов.

### Experimental BGE-M3 profile

Для debug/исследовательского профиля используйте:

```powershell
pwsh -File scripts/agents/ensure_repo_semantic_search.ps1 `
  -Build `
  -Profile gpu-bge-m3 `
  -TargetRepoPath C:\nullexp_vpn\vpn-server
```

Этот профиль:
- model: `BAAI/bge-m3`
- profile: `gpu_bge_m3`

Он не считается официальным fallback-путём, потому что cold-start для него
слишком дорогой. Если `Qwen3` сломан, рекомендуемый operational fallback:
- перейти на `-Profile cpu`;
- либо чинить `Qwen3`, а не маскировать проблему тяжёлым GPU fallback.

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

Для E5-family в CPU default query/document formatting обязателен:
- query: `query: {query}`
- documents: `passage: {text}`

`fastembed_local` остаётся как fallback/dev-режим, но не является основным
production-like профилем.

GPU profile сознательно вынесен отдельно:
- CPU default остаётся дешёвым и стабильным после ребута;
- основной GPU profile на `Qwen3` даёт лучший текущий retrieval result;
- `BAAI/bge-m3` сохраняется только как experimental/debug profile.

## Isolation rule

Коллекции Qdrant теперь обязаны различаться не только по модели, но и по профилю.
Это нужно, чтобы:
- не смешивать CPU E5 и GPU embeddings;
- не смешивать профили с разным query/document formatting;
- безопасно сравнивать несколько моделей параллельно.

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
pwsh -File scripts/agents/ensure_repo_semantic_search.ps1 -Profile cpu -TargetRepoPath C:\nullexp_vpn\vpn-server
```

Для GPU-профиля:

```powershell
pwsh -File scripts/agents/ensure_repo_semantic_search.ps1 -Profile gpu -TargetRepoPath C:\nullexp_vpn\vpn-server
```
