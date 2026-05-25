# Спецификация: Repo Semantic Search Lifecycle And Backend Abstraction

> **Версия:** 1.0
> **Дата:** 2026-04-20
> **Статус:** Draft
> **Основание:** product redesign после bootstrap/runtime regressions и перехода от single-repo runtime к multi-repo persistent service
> **Связанные документы:**
> - `docs/specifications/P1.Z-repo-semantic-search-mcp-specification.md`
> - `docs/strategy/P1.Y-EMBEDDING-PROFILES-AND-MODEL-STRATEGY.md`
> - `docs/migration/REPO-SEMANTIC-SEARCH-MIGRATION-MAP.md`

---

## 0) Контекст и цель

`repo-semantic-search` уже существует как работающий semantic MCP, но текущий operational contract
оказался продуктово неверным для daily agent workflow.

Подтверждённые проблемы текущего состояния:

1. Runtime фактически привязан к **одному активному репозиторию**.
2. Переключение на другой repo может приводить к purge/retarget behavior вместо reuse готового индекса.
3. Bootstrap MCP-клиента может запускать тяжёлую работу:
   - retarget active repo;
   - full rebuild;
   - dependency startup.
4. Watcher lifecycle смешан с bootstrap и indexing lifecycle.
5. Inference runtime слишком близко привязан к деталям машины:
   - Docker GPU;
   - WSL native server;
   - host-local helpers;
   - внешние соседние проекты.

Это ломает главный user expectation:

- агент должен быстро подключиться к MCP;
- увидеть готовый статус;
- искать по уже построенному индексу;
- а не ждать скрытый cold start в момент первого запроса.

### Цель этой спецификации

Перевести `repo-semantic-search` к следующей продуктовой модели:

1. **Множественные persistent indexes**, по одному на каждый репозиторий.
2. **Явный lifecycle** для repo attach / build / activate / watcher.
3. **Никакого implicit indexing на client connect**.
4. **Один active watcher** только для active repo.
5. **Inference backend abstraction**, чтобы MCP работал с backend roles/capabilities, а не с деталями железа и способа запуска.

---

## 1) Продуктовая формулировка

`repo-semantic-search` — это локальный semantic index service для нескольких репозиториев,
с persistent per-repo indexes, explicit lifecycle, единым active watcher и pluggable inference backends.

Сервис должен давать три гарантийных свойства:

1. **Repo ownership**
   Каждый индекс принадлежит конкретному `repo_root` и не смешивается с другими репозиториями.

2. **Fast bootstrap**
   Подключение MCP-клиента не запускает тяжёлые side effects.

3. **Explicit lifecycle**
   Тяжёлые действия выполняются только явными lifecycle tools/commands.

---

## 2) Утверждённые решения

1. **Single-repo runtime model отменяется.**
   Сервис должен хранить несколько готовых индексов одновременно.

2. **Source of truth для repo lifecycle state выносится в отдельный registry layer.**
   Векторы и payload остаются в `Qdrant`, но lifecycle metadata не должен жить только в именах коллекций или в runtime памяти.

3. **Registry layer реализуется на `SQLite`.**
   Причина:
   - нужен локальный persistent metadata store;
   - нужен cheap read/write lifecycle state;
   - нужен один writer/локальный daemon, без требований к distributed consensus;
   - это проще и надёжнее, чем JSON-файл и чище, чем перегружать Qdrant ролью registry.

4. **`Qdrant` остаётся vector store, а не lifecycle registry.**

5. **Watcher работает только для одного active repo.**
   Multi-watcher mode не входит в target contract v2.

6. **Bootstrap MCP transport не имеет права запускать full rebuild.**
   Максимум:
   - поднять лёгкий runtime/status layer;
   - вернуть fast-fail status;
   - запустить already-configured lightweight process, если он не требует rebuild.

7. **Индексация выполняется только explicit lifecycle action.**
   Нормативные операции:
   - `build_index(repo_root)`
   - `rebuild_index(repo_root)`
   - `reindex_paths(repo_root, paths)`

8. **`activate_repo(repo_root)` не запускает rebuild автоматически.**
   Если индекс отсутствует или stale, сервис должен вернуть это как статус, а не скрыто чинить.

9. **Inference runtime абстрагируется через backend registry.**
   MCP и index lifecycle должны работать с backend roles:
   - embedding
   - reranker
   - colbert
   - optional llm helper
   а не с hardware-specific branching.

10. **Index profile и inference backend — разные сущности.**
    Они не должны смешиваться.

11. **Auto-detection include globs / doc prefixes разрешён только как initial proposal.**
    После первого attach/build эти значения сохраняются в repo registry и считаются частью repo-specific config.

12. **Client connect никогда не меняет active repo автоматически.**

---

## 3) Scope / Non-goals

### Входит в scope

- Per-repo persistent index registry.
- SQLite metadata store.
- Active repo model.
- Single active watcher model.
- Explicit repo lifecycle tools/commands.
- Backend abstraction for embedding/reranker/colbert roles.
- Fast-fail bootstrap/status contract.
- Migration path from current single-repo operational model.

### Не входит в scope

- Полноценный multi-user cloud control plane.
- Federated distributed search по нескольким репозиториям в одном запросе.
- LLM answer synthesis.
- Full RAG orchestration поверх retrieval.
- Background build scheduler с очередями и worker farm.
- Несколько watchers одновременно.

---

## 4) Ключевые сущности

### 4.1 Repo registry entry

Для каждого зарегистрированного репозитория сервис хранит entry со следующими полями:

- `repo_root`
- `repo_key`
- `display_name`
- `status`
- `active`
- `index_profile`
- `include_globs`
- `doc_prefixes`
- `exclude_globs`
- `last_full_build_ts`
- `last_incremental_update_ts`
- `last_error`
- `watch_enabled`
- `watch_running`
- `code_points_count`
- `docs_points_count`
- `created_at`
- `updated_at`

### 4.2 Repo lifecycle status

Допустимые состояния:

- `registered`
- `indexed`
- `stale`
- `indexing`
- `error`
- `disabled`

### 4.3 Active repo session

Отдельная runtime-сущность:

- в один момент времени существует не более одного `active_repo`;
- watcher привязан только к `active_repo`;
- search tools по умолчанию работают против `active_repo`, если явно не указан другой repo.

### 4.4 Index profile

`Index profile` описывает совместимость stored dense vectors.

Он включает:

- embedding model identity;
- query template;
- document prefix;
- prompt contract;
- schema/version compatibility.

Примеры:

- `cpu_e5`
- `gpu_qwen3`
- `gpu_pplx`

### 4.5 Inference backend

`Inference backend` описывает runtime endpoint/capabilities, через который выполняется инференс.

Он включает:

- `backend_id`
- `role`
- `transport`
- `base_url` или local provider config
- `backend_type`
- `location_type`
- `device_type`
- `managed_by_service`
- `autostart_policy`
- `healthcheck_contract`
- `capabilities`

Примеры:

- `embedding/wsl_local_pplx`
- `embedding/docker_cpu_tei`
- `reranker/wsl_local_qwen3`
- `colbert/host_http_jina`

Для env/template слоя:

- checked-in `*.example` не должны содержать machine-local WSL/host/model paths;
- локальные значения обязаны жить в ignored override files
  (`.env.cpu`, `.env.gpu`, `.env.gpu.qwen3`, `.env.gpu.bge-m3`, `.env*.local`);
- helper/deploy orchestration обязан явно резолвить base env + local override layers,
  а не полагаться на неявный порядок файлов или ручное переписывание examples.

---

## 5) Разделение ответственности

Новая архитектура обязана явно разделить следующие контуры:

1. **MCP transport**
   - stdio / streamable-http
   - tools/resources exposure
   - bootstrap/status endpoints

2. **Repo registry**
   - хранение metadata per repo
   - active repo state
   - lifecycle status

3. **Index lifecycle**
   - attach/register
   - build/rebuild
   - partial reindex
   - mark stale / clear error

4. **Watcher lifecycle**
   - start
   - stop
   - active repo switch
   - incremental updates only

5. **Inference backend management**
   - backend selection
   - backend health
   - backend autostart policy
   - backend role routing

Смешивание этих обязанностей в одном bootstrap path запрещено.

---

## 6) Целевой lifecycle

### 6.1 Register / attach

Операция:

- `register_repo(repo_root, options?)`

Что делает:

- нормализует `repo_root`;
- вычисляет `repo_key`;
- строит initial proposal:
  - `include_globs`
  - `doc_prefixes`
  - `exclude_globs`
- сохраняет registry entry;
- не строит индекс;
- не делает repo active автоматически, если это не указано явно.

Что не делает:

- не индексирует;
- не стартует watcher;
- не запускает inference backends.

### 6.2 Build / rebuild

Операции:

- `build_index(repo_root)`
- `rebuild_index(repo_root)`

Что делают:

- выбирают index profile;
- выбирают backend для embedding role;
- строят/обновляют collections в Qdrant;
- обновляют counts/timestamps/status в SQLite.

Что обязаны гарантировать:

- статус `indexing` виден во время сборки;
- search по repo до `ready/indexed` не выполняется как будто всё готово;
- никакая client session не должна путать build с bootstrap.

### 6.3 Activate repo

Операция:

- `activate_repo(repo_root)`

Что делает:

- меняет `active_repo` в registry/runtime;
- при необходимости останавливает watcher старого active repo;
- может стартовать watcher нового active repo только если:
  - repo already indexed;
  - watcher policy это разрешает.

Что не делает:

- не rebuild-ит автоматически;
- не purge-ит другие repo indexes;
- не retarget-ит transport side-effect’ом.

### 6.4 Start / stop watcher

Операции:

- `start_watcher()`
- `stop_watcher()`

Что делают:

- работают только для `active_repo`;
- watcher делает incremental sync;
- watcher не запускает full rebuild.

### 6.5 Search

Операции поиска:

- `semantic_search`
- `semantic_search_code`
- `semantic_search_docs`
- `hybrid_search`

По умолчанию работают против:

- `active_repo`, если repo явно не указан.

Если `active_repo`:

- не установлен;
- не indexed;
- stale/error;

сервис должен fast-fail с явным status/error payload.

---

## 7) Что запрещено на client connect

Bootstrap MCP-клиента не должен:

1. менять `active_repo`;
2. запускать full build;
3. запускать implicit retarget;
4. purge-ить collections другого repo;
5. запускать expensive dependency startup, если это не лёгкий managed background process без rebuild;
6. блокировать подключение до завершения полного indexing.

Допустимое поведение bootstrap:

- вернуть status;
- отдать список repos;
- показать active repo;
- показать search availability;
- сообщить, что repo not indexed / stale / indexing.

---

## 8) Inference backend abstraction

### 8.1 Почему это нужно

На разных машинах backend стартует по-разному:

- CPU модель в Docker;
- GPU TEI в Docker;
- WSL-native PyTorch server;
- host-local HTTP process;
- отдельный remote service.

MCP не должен содержать product logic уровня:

- “если Windows, делай WSL hack”;
- “если Docker GPU сломан, переключи путь”;
- “если соседний проект поднял embedder, используем его”.

Это должно жить в backend registry/config.

### 8.2 Роли backend’ов

Минимально v2 должен поддерживать роли:

- `embedding`
- `reranker`
- `colbert`

Для `repo-semantic-search` v2 обязательной является только роль `embedding`.
Остальные роли входят в архитектурный scope, но могут не использоваться в первом implementation slice.

### 8.3 Типы backend’ов

Поддерживаемые типы:

- `tei_http`
- `openai_embeddings_http`
- `local_fastembed`
- `custom_http`

Новая реализация должна проектироваться так, чтобы добавить новый backend type
не требовало переписывания repo lifecycle.

### 8.4 Backend registry

В registry нужно хранить:

- `backend_id`
- `role`
- `backend_type`
- `transport`
- `endpoint/base_url`
- `managed_by_service`
- `autostart_policy`
- `health_path` / health contract
- `location_type` (`docker`, `wsl`, `host`, `remote`)
- `device_type` (`cpu`, `gpu`)
- `config_blob`

### 8.5 Автостарт backend’ов

Нормативное правило:

- explicit build command может запускать managed backend для нужной роли;
- client bootstrap не должен запускать тяжёлый backend cold start скрыто;
- если backend не готов, lifecycle/build должен показывать это явно.

---

## 9) Storage design

### 9.1 Qdrant

`Qdrant` хранит:

- dense vectors;
- payload metadata;
- per-scope collections:
  - `code`
  - `docs`

Коллекции должны включать:

- repo key;
- profile slug;
- model slug;
- scope;
- schema version.

### 9.2 SQLite

`SQLite` хранит:

- repo registry;
- active repo state;
- lifecycle state;
- watcher state;
- backend registry;
- repo/backend bindings;
- timestamps and errors.

### 9.3 Why SQLite + Qdrant

Такое разделение даёт:

- дешёвое lifecycle state управление;
- ясный source of truth для active repo;
- отсутствие перегрузки Qdrant метаданными orchestration уровня;
- возможность быстрых status/list queries без похода в vector store как в базу продукта.

---

## 10) Watcher model

### 10.1 Target behavior

Watcher:

- существует максимум один;
- привязан к active repo;
- занимается только incremental updates;
- обновляет registry timestamps/status.

### 10.2 Repo switch behavior

При `activate_repo(new_repo)`:

1. current watcher stop;
2. active repo pointer updates;
3. если `new_repo` indexed и watcher policy enabled:
   - watcher start against new repo;
4. если `new_repo` not indexed:
   - watcher не стартует;
   - статус это явно отражает.

### 10.3 Non-goals for watcher

Watcher не должен:

- строить индекс с нуля;
- silently fix incompatible profile;
- пересоздавать весь repo index;
- одновременно наблюдать несколько репозиториев.

---

## 11) Repo scope detection

### 11.1 Initial proposal

При `register_repo` разрешён auto-detect:

- `include_globs`
- `doc_prefixes`

### 11.2 Persisted config

После attach/build эти значения сохраняются как repo-specific config.

Это значит:

- auto-detect используется только как initial proposal;
- далее lifecycle tools работают уже с сохранённой конфигурацией;
- client connect не должен заново изобретать scope каждый раз.

### 11.3 UX expectation

Status tools должны показывать:

- effective include globs;
- effective doc prefixes;
- источник значения:
  - `auto-proposed`
  - `user-overridden`

---

## 12) MCP tool contract v2

### 12.1 Registry / lifecycle tools

Минимальный новый набор:

- `list_repos()`
- `list_legacy_repo_entries()`
- `get_repo_status(repo_root?)`
- `register_repo(repo_root, include_globs?, doc_prefixes?, profile?)`
- `activate_repo(repo_root)`
- `build_index(repo_root, force_rebuild=false)`
- `reindex_paths(repo_root, paths)`
- `start_watcher(repo_root?)`
- `stop_watcher()`
- `prune_legacy_repo_entries(force_remove_active=false)`

### 12.2 Search tools

Существующие search tools сохраняются, но контракт уточняется:

- поиск не должен implicitly индексировать repo;
- если repo не готов, tool возвращает явную ошибку lifecycle уровня.

### 12.3 Backend tools

Минимальный backend набор:

- `list_backends()`
- `get_backend_status(role or backend_id)`
- `set_role_backend(role, backend_id)`
- `get_backend_action_plan(role or backend_id, repo_root?)`
- `ensure_backend(role or backend_id, repo_root?)`

Важно:

- backend lifecycle tools не делают hidden startup из MCP runtime;
- `ensure_backend` возвращает explicit verdict и host-side next action;
- реальный start/switch остаётся host-side через repo-owned helper scripts.

---

## 13) Status contract

`index_status` / `repo_status` должны возвращать:

- `repo_root`
- `repo_key`
- `repo_state`
- `active`
- `search_available`
- `reason_if_unavailable`
- `index_profile`
- `embedding_backend`
- `embedding_backend_id`
- `selected_embedding_backend_id`
- `backend_switch_required`
- `watch_enabled`
- `watch_running`
- `last_full_build_ts`
- `last_incremental_update_ts`
- `collections`
- `counts code/docs`
- `include_globs`
- `doc_prefixes`

Если repo не готов:

- `search_available = false`
- `reason_if_unavailable` обязан быть явным:
  - `repo_not_registered`
  - `index_missing`
  - `indexing_in_progress`
  - `backend_unhealthy`
  - `index_error`

---

## 14) Migration strategy

### 14.1 Current model to be retired

Нужно вывести из target contract следующие behaviors:

- one-running-stack-one-repo semantics как user-facing behavior;
- implicit retarget on connect;
- purge stale collections как способ “переключиться” между repo;
- hidden rebuild during Codex bootstrap.

### 14.2 Safe transition

Переход рекомендуется делать по слоям:

1. ввести SQLite registry;
2. отделить active repo и lifecycle status от runtime памяти;
3. выключить implicit indexing on connect;
4. перевести search tools на fast-fail lifecycle-aware contract;
5. перевести helper/deploy orchestration на explicit backend catalog / registry contract;
6. добавить cold-boot acceptance smoke, который проходит через helper + MCP tools;
7. только потом расширять watcher switching и дополнительные backend roles.

Для старых registry entries вида `/repo` и `/target_repo` migration cleanup должен
оставаться explicit:

- `list_legacy_repo_entries()`
- `prune_legacy_repo_entries(force_remove_active=false)`

---

## 15) Implementation slices

### Slice A. Registry foundation

Deliverables:

- SQLite schema;
- repo registry layer;
- active repo state;
- status APIs.

### Slice B. Bootstrap hardening

Deliverables:

- client bootstrap no longer triggers rebuild;
- fast-fail status on missing/stale index;
- explicit lifecycle commands only.

### Slice C. Multi-repo Qdrant persistence

Deliverables:

- multiple repo collections coexist;
- no purge on repo switch;
- repo-aware collection lookup.

### Slice D. Active watcher lifecycle

Deliverables:

- single watcher bound to active repo;
- watcher switching on activate;
- no rebuild from watcher start.

### Slice E. Backend abstraction foundation

Deliverables:

- backend registry schema;
- role -> backend mapping;
- embedding backend decoupled from machine-specific logic.

### Slice F. Repo scope config persistence

Deliverables:

- include/doc prefixes saved per repo;
- lifecycle tools expose and update them;
- autodetect only as initial proposal.

---

## 16) Acceptance criteria

### Product acceptance

1. На машине можно хранить готовые индексы минимум для двух разных репозиториев одновременно.
2. Переключение active repo не удаляет чужой индекс.
3. Новый MCP bootstrap не запускает full build неявно.
4. Search tool against unindexed repo fast-fails с явным статусом.
5. Watcher работает только для active repo.
6. Active repo switch корректно переводит watcher на новый repo или не стартует его, если индекс не готов.
7. Repo scope detection виден и сохраняется как repo config.
8. Runtime path поддерживает backend abstraction, а не machine-specific if/else как главный контракт.

### Operational acceptance

1. `index_status` отвечает быстро и не зависит от полного rebuild.
2. `list_repos` показывает registry без похода в full indexing path.
3. Если embedding backend unhealthy, причина видна в status/error, а не маскируется в generic timeout.
4. Cold start build выполняется только explicit командой и может занимать долго, не ломая bootstrap.

---

## 17) Open architectural choice

Единственный сознательно отложенный вопрос:

- нужно ли v2 search tools позволять явно указывать `repo_root` в запросе,
  или они обязаны всегда работать только через `active_repo`.

Рекомендуемое решение для first rollout:

- по умолчанию search работает через `active_repo`;
- optional explicit `repo_root` можно добавить позже как расширение,
  когда registry/status contract стабилизируется.

---

## 18) Нормативный итог

После реализации этой спецификации `repo-semantic-search` перестаёт быть
“MCP, который иногда сам чинит runtime и индекс по дороге”.

Он становится:

- multi-repo persistent semantic service;
- с explicit lifecycle;
- с predictable bootstrap;
- с single active watcher;
- и с backend abstraction, пригодной для дальнейшего роста в более широкий retrieval/RAG contour.
