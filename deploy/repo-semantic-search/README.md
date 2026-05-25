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

## Поддержка платформ

- Windows: основной documented path
- Ubuntu/Linux: поддерживается
- macOS: поддерживается только CPU profile

GPU путь рассчитан на:
- Windows с рабочим Docker GPU passthrough
- Linux с `NVIDIA Container Toolkit`

На macOS использовать только `-Profile cpu`.

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

Рекомендуемый user-facing entrypoint:

Windows:

```powershell
pwsh -File scripts/agents/start_repo_semantic_for_project.ps1 `
  -RepoPath C:\path\to\target-repo
```

Linux/macOS:

```bash
bash scripts/agents/start_repo_semantic_for_project.sh \
  --repo-path /path/to/target-repo
```

`ensure_repo_semantic_search.*` остаётся низкоуровневым helper script, если нужен
явный контроль над `-Build`, `-Clean`, `-ResetVolumes` или `-Profile`.

Если нужен прямой вызов helper script, явно укажите target repo, который нужно индексировать.

```powershell
pwsh -File scripts/agents/ensure_repo_semantic_search.ps1 `
  -Build `
  -Profile cpu `
  -TargetRepoPath C:\path\to\target-repo
```

Linux/macOS shell-вариант:

```bash
bash scripts/agents/ensure_repo_semantic_search.sh \
  --build \
  --profile cpu \
  --target-repo-path /path/to/target-repo
```

Для ручного compose-запуска:

```bash
cd deploy/repo-semantic-search
docker compose -f docker-compose.repo-semantic-search.yml --env-file .env.example up -d --build
```

После старта helper script ждёт host-visible `/readyz`, а не только состояние контейнера или внутренний loopback в самом контейнере.

`-Clean` останавливает compose project и убирает orphan containers, но сохраняет persistent volumes.
Для несовместимых storage/schema миграций используйте явный `-ResetVolumes` вместе с `-Clean`.

Важно: один запущенный стек индексирует один target repo. Для переключения на другой
репозиторий нужно перезапустить тот же стек с новым `-TargetRepoPath`.

При таком host-side switch helper также активирует текущий logical repo в lifecycle registry.
Это значит:
- текущий runtime после `ensure_repo_semantic_search.* -TargetRepoPath ...` становится `active`;
- cross-repo `activate_repo()` и `build_index()` внутри MCP для другого repo не делают скрытый switch,
  а возвращают `requires_runtime_switch=true`.

Коллекции теперь включают repo-specific key, поэтому два разных target repo
не смогут тихо делить один и тот же индекс.

Если в registry остались legacy entries старого формата (`/repo`, `/target_repo`),
их cleanup нужно делать явно, а не скрыто:

- `list_legacy_repo_entries()`
- `prune_legacy_repo_entries()`

Это позволяет убрать старые placeholder-записи после migration, не задевая текущий active repo.

## GPU profile

Для машины с NVIDIA GPU используйте primary GPU profile:

```powershell
pwsh -File scripts/agents/ensure_repo_semantic_search.ps1 `
  -Build `
  -Profile gpu `
  -TargetRepoPath C:\path\to\target-repo
```

Linux shell-вариант:

```bash
bash scripts/agents/ensure_repo_semantic_search.sh \
  --build \
  --profile gpu \
  --target-repo-path /path/to/target-repo
```

Этот профиль теперь основной:
- model: `Qwen/Qwen3-Embedding-0.6B`
- profile: `gpu_qwen3`
- backend: `embedding/gpu_qwen3_bundled_tei`
- launch mode: bundled Docker compose (`qdrant` + `tei` + `repo-semantic-mcp`)
- host TEI health endpoint: `http://127.0.0.1:8084/health`

`-Profile gpu` больше не запускает repo-owned WSL/native `repo_semantic_gpu_server.py`.
Если нужен старый managed-host backend, его нужно выбирать явно через env/backend id,
а не через primary recovery path.

Для выбора конкретной GPU используйте stable UUID из:

```powershell
nvidia-smi -L
```

Значение прописывается в ignored machine-local env:

```dotenv
SEMANTIC_MCP_NVIDIA_VISIBLE_DEVICES=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
SEMANTIC_MCP_CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

На машине с RTX 5060 Ti + Tesla V100 primary local config должен указывать V100 UUID,
а не индекс `0`/`1`, потому что порядок GPU может измениться после driver/Docker changes.

Для RTX 50xx / Blackwell профиль должен использовать architecture-specific TEI image:
- `ghcr.io/huggingface/text-embeddings-inference:120-1.9`

Warmup также ограничен более реалистичным batching budget:
- `SEMANTIC_MCP_TEI_MAX_BATCH_TOKENS=4096`
- `SEMANTIC_MCP_TEI_MAX_CONCURRENT_REQUESTS=64`

Это сделано специально, чтобы не платить дефолтный жирный GPU warmup на `16384` токенов
при локальном repo-search workload.

На текущей машине после перехода на `120-1.9` и более дешёвого warmup budget
cold start этого профиля до готового MCP занимает примерно:
- `~2.5 мин` для первого clean start после смены image/runtime;
- `~1.5 мин` для повторного clean restart на том же профиле.

То есть bottleneck теперь сидит в TEI/Qwen3 warmup, но уже не выглядит как
неприемлемые `7-8 минут`.

Эквивалентный ручной запуск:

```bash
cd deploy/repo-semantic-search
docker compose -f docker-compose.repo-semantic-search.yml -f docker-compose.repo-semantic-search.gpu.yml --env-file .env.gpu.example up -d --build
```

Профиль рассчитан на `NVIDIA Container Toolkit` и TEI image, соответствующий compute capability GPU.
Для RTX 50xx используйте ветку `120-1.9`, а не generic `cuda-*` tag.
Если выбранная GPU не поддерживает image/model/runtime, фиксируйте это как backend blocker
и переключайтесь на другой GPU/image только явно.

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
  -TargetRepoPath C:\path\to\target-repo
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
- Compose default использует `qdrant/qdrant:v1.17.1`; это нужно для текущего
  named dense+sparse vector schema. Если меняете image, сначала проверьте
  совместимость с `SEMANTIC_MCP_INDEX_SCHEMA_VERSION`.
- `SQLite registry` lifecycle-метаданных также должен жить на persistent volume;
  в compose для этого используется `repo_semantic_registry_data`.
- `TEI` также должен жить с persistent cache volume, иначе после очистки docker volume
  модель будет скачиваться заново.
- Для RTX 50xx / Blackwell не используйте generic `cuda-*` image как основной путь.
  Практический default для этого класса карт: `120-1.9`.
- `readyz` означает готовность transport/runtime слоя, а не наличие построенного индекса.
- Поисковый индекс строится только explicit lifecycle-инструментами (`build_index`, `rebuild_index`);
  bootstrap и client connect не запускают rebuild автоматически.
- `SEMANTIC_MCP_WATCH_ENABLED=1` означает, что runtime поддерживает watcher,
  но не гарантирует, что polling thread сейчас запущен. Проверяйте
  `index_status().watcher.running`; при `watcher.start_required=true` live
  incremental indexing начнется только после явного `start_watcher`.
- Явный `start_watcher` сначала делает startup reconcile файлов, измененных пока
  watcher был остановлен, а затем запускает polling для будущих изменений. Это
  не full rebuild и не выполняется из status/search автоматически.
- Backend role binding хранится отдельно в SQLite registry:
  - `list_backends()`
  - `get_backend_status(role|backend_id)`
  - `set_role_backend(role, backend_id)`
  - `get_backend_action_plan(role|backend_id, repo_root?)`
  - `ensure_backend(role|backend_id, repo_root?)`
- Host-side helper orchestration теперь использует repo-owned backend catalog
  (`services/repo_semantic/backend_catalog.py` + `scripts/runtime/repo_semantic_backend_catalog.py`).
  Выбор launch path (`bundled_compose | managed_host | external_manual`) идёт по `SEMANTIC_MCP_EMBEDDING_BACKEND_ID`,
  а не по эвристике `TEI_URL` или скрытому порядку `.env*`.
- Checked-in `*.example` env files больше не должны хранить machine-local значения.
  Используйте ignored local overrides:
  - `deploy/repo-semantic-search/.env.cpu`
  - `deploy/repo-semantic-search/.env.gpu`
  - `deploy/repo-semantic-search/.env.gpu.qwen3`
  - `deploy/repo-semantic-search/.env.gpu.bge-m3`
  - `deploy/repo-semantic-search/.env*.local`
- Helper резолвит base env и local override layers явно. Для GPU Docker path
  `.env.gpu` или `.env.gpu.local` могут хранить machine-local GPU UUID.
- Если выбранный backend для роли `embedding` не совпадает с текущим runtime backend,
  `build_index` и `start_watcher` fast-fail’ятся с `backend_switch_required=true`
  вместо скрытого hot-switch.
- `get_backend_action_plan` и `ensure_backend` не запускают Docker/PowerShell из контейнера:
  они возвращают explicit operator verdict (`ready | runtime_switch_required | manual_backend_start_required | backend_unhealthy`)
  и host-side `ensure_repo_semantic_search.ps1` command hint, если нужно действие снаружи.
- Optional roles `reranker` и `colbert` уже живут в том же backend registry/action-plan слое.
  Их можно seed'ить через env contract (`SEMANTIC_MCP_RERANKER_*`, `SEMANTIC_MCP_COLBERT_*`)
  без hidden runtime wiring в connect-path.
- Если используется `tei_http`, runtime работает с HTTP endpoint из `SEMANTIC_MCP_TEI_URL`.
- Базовый compose по-прежнему содержит bundled `tei`, но repo-owned GPU path может использовать внешний backend;
  это должно быть видно в backend registry/status, а не угадываться по implicit rebuild или connect-path.

## Регистрация в Codex и Claude

После первого успешного запуска можно зарегистрировать локальные agent integrations:

```powershell
pwsh -File scripts/agents/register_repo_semantic_search.ps1
```

Скрипт обновляет:
- `%USERPROFILE%\.codex\config.toml`
- `%USERPROFILE%\.claude.json`

Поведение после регистрации:
- Codex использует host-side `stdio` wrapper, который проксирует MCP JSON-RPC в shared HTTP endpoint;
- wrapper может стартовать уже существующий контейнер, но не делает build/rebuild/switch;
- restart/rebuild backend container больше не должен убивать transport текущего чата;
- Claude сохраняет shared HTTP URL-based integration;
- оба клиента продолжают использовать один и тот же target repo, profile и Qdrant index state.

Кроссплатформенный вариант:

```bash
python3 scripts/agents/register_repo_semantic_search.py
```

## Поведение после ребута машины

- Все контейнеры стека работают с `restart: unless-stopped`.
- Если Docker Desktop стартует вместе с Windows, semantic MCP поднимется автоматически.
- Если Docker Desktop был остановлен вручную или нужно перепроверить готовность,
  используйте:

```powershell
pwsh -File scripts/agents/ensure_repo_semantic_search.ps1 -Profile cpu -TargetRepoPath C:\path\to\target-repo
```

Для GPU-профиля:

```powershell
pwsh -File scripts/agents/ensure_repo_semantic_search.ps1 -Profile gpu -TargetRepoPath C:\path\to\target-repo
```

Для Qwen3-каталога:

```powershell
pwsh -File scripts/agents/ensure_repo_semantic_search.ps1 -Profile gpu-qwen3 -TargetRepoPath C:\path\to\target-repo
```

Для холодного smoke после ребута:

```powershell
pwsh -File scripts/agents/smoke_repo_semantic_boot.ps1 -Profile gpu -RepoPath C:\path\to\repo-semantic-mcp
```

Smoke делает explicit helper-start, затем через MCP проверяет:
- `index_status`
- `get_backend_status(role="embedding")`
- `ensure_backend(role="embedding")`
- один реальный `semantic_search`

Для агентского readiness gate без lifecycle side effects:

```powershell
py -3.12 scripts/agents/repo_semantic_ensure_ready.py --repo C:\path\to\target-repo
```

Этот helper по умолчанию работает в read-only `diagnose` mode. Если нужно
разрешить bounded repair без rebuild, явно используйте:

```powershell
py -3.12 scripts/agents/repo_semantic_ensure_ready.py --repo C:\path\to\target-repo --mode safe-recover
```

`safe-recover` может стартовать watcher для уже активного совпадающего repo/profile,
но не запускает `build_index`, `rebuild_index`, `build_graph` или `rebuild_graph`.
Прямые fallback-вызовы через `repo_semantic_call_tool.py` возвращают стабильный
`ok/error` envelope по умолчанию; `--payload` оставлен только для ручной отладки.

Если нужен shell-вариант:

```bash
PROFILE=gpu REPO_PATH=/path/to/repo-semantic-mcp bash scripts/agents/smoke_repo_semantic_boot.sh
```
