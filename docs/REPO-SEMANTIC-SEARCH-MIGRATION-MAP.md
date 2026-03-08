# Repo Semantic Search: Migration Map

> **Дата:** 2026-03-08
> **Статус:** Working draft
> **Назначение:** ручной перенос semantic MCP из исходного private monorepo в отдельный репозиторий без потери рабочего состояния

---

## 1) Цель миграции

Вынести semantic MCP в отдельный reusable репозиторий, чтобы:
- развивать его независимо от исходного private monorepo;
- добавить quality upgrades вне рамок текущего monorepo;
- подготовить отдельный GitHub-репозиторий и делиться им с коллегами;
- не тащить в generic инструмент доменные особенности исходного private monorepo.

---

## 2) Рекомендуемое имя нового проекта

Рекомендуемый путь:

`C:\cursor_mcp\repo-semantic-search`

Причина выбора:
- имя уже совпадает с текущим runtime/compose/server name;
- не слишком абстрактное;
- не завязано на исходный private monorepo;
- удобно для GitHub-репозитория и install/run docs.

---

## 3) Принцип миграции

### Фаза 1: safe landing

Переносим рабочий контур почти без реорганизации layout.

Цель:
- сначала получить в новом каталоге тот же работающий runtime;
- только потом чистить структуру и обобщать tool.

### Фаза 2: cleanup and generalization

После успешного старта в новом проекте:
- убираем repo-specific доменные особенности;
- вводим profile system;
- подготавливаем GPU profile;
- делаем GitHub-ready документацию.

Это важно: **не смешивать перенос и большой рефакторинг в один шаг**.

---

## 4) Что переносить в новую репу сейчас

### 4.1 Код сервиса

Перенести как есть:

- `apps/repo-semantic-mcp/`
- `services/repo_semantic/`

### 4.2 Deploy / runtime

Перенести как есть:

- `deploy/repo-semantic-search/`

### 4.3 Helper scripts

Перенести:

- `scripts/agents/ensure_repo_semantic_search.ps1`
- `scripts/agents/register_repo_semantic_search.ps1`

### 4.4 Документация

Перенести как стартовую базу:

- `docs/specifications/platform/P1.Z-repo-semantic-search-mcp-specification.md`
- `deploy/repo-semantic-search/README.md`
- этот файл

---

## 5) Что не переносить как есть

### 5.1 Repo-specific agent layer

Не переносить:

- `AGENTS.md`
- `CLAUDE.md`
- `agent_context/**`
- `docs/guides/AGENT-CONTEXT-MAINTENANCE.md`

Причина:
- это слой инструкций именно для исходного private monorepo;
- в новом репо нужен уже свой агентский bootstrap.

### 5.2 Unrelated scripts

Не переносить:

- `scripts/agents/check_repo_skills.py`
- `scripts/agents/install_repo_skills.py`
- `scripts/agents/sync_repo_skills.py`

Причина:
- это tooling для текущего monorepo и его agent-context.

### 5.3 Доменные документы source monorepo

Не переносить как runtime docs:

- Telegram/VPN-flow specs
- architecture docs
- maintenance/user-flow документы

Их можно использовать только как историческое обоснование дизайна.

---

## 6) Целевая структура для safe landing

На первом шаге рекомендован такой layout в новом проекте:

- `apps/repo-semantic-mcp/`
- `services/repo_semantic/`
- `deploy/repo-semantic-search/`
- `scripts/agents/`
- `docs/specifications/`

Почему не делать сразу `src/semantic_mcp/`:
- это добавит лишний рефакторинг в момент переноса;
- сначала важнее сохранить working runtime;
- rename/repackage можно сделать уже после первого успешного старта в новой репе.

---

## 7) Карта переноса: source -> target

### 7.1 App entrypoint

- `apps/repo-semantic-mcp/main.py`
  -> `C:\cursor_mcp\repo-semantic-search\apps\repo-semantic-mcp\main.py`

- `apps/repo-semantic-mcp/requirements.txt`
  -> `C:\cursor_mcp\repo-semantic-search\apps\repo-semantic-mcp\requirements.txt`

- `apps/repo-semantic-mcp/Dockerfile`
  -> `C:\cursor_mcp\repo-semantic-search\apps\repo-semantic-mcp\Dockerfile`

### 7.2 Core package

- `services/repo_semantic/__init__.py`
  -> `C:\cursor_mcp\repo-semantic-search\services\repo_semantic\__init__.py`
- `services/repo_semantic/config.py`
  -> `...\services\repo_semantic\config.py`
- `services/repo_semantic/indexer.py`
  -> `...\services\repo_semantic\indexer.py`
- `services/repo_semantic/logging.py`
  -> `...\services\repo_semantic\logging.py`
- `services/repo_semantic/mcp_server.py`
  -> `...\services\repo_semantic\mcp_server.py`
- `services/repo_semantic/models.py`
  -> `...\services\repo_semantic\models.py`
- `services/repo_semantic/qdrant_store.py`
  -> `...\services\repo_semantic\qdrant_store.py`
- `services/repo_semantic/search_service.py`
  -> `...\services\repo_semantic\search_service.py`
- `services/repo_semantic/watcher.py`
  -> `...\services\repo_semantic\watcher.py`

### 7.3 Chunkers

- `services/repo_semantic/chunkers/__init__.py`
- `services/repo_semantic/chunkers/factory.py`
- `services/repo_semantic/chunkers/generic.py`
- `services/repo_semantic/chunkers/markdown.py`
- `services/repo_semantic/chunkers/python.py`

Перенести в:

- `C:\cursor_mcp\repo-semantic-search\services\repo_semantic\chunkers\*`

### 7.4 Embeddings

- `services/repo_semantic/embeddings/__init__.py`
- `services/repo_semantic/embeddings/base.py`
- `services/repo_semantic/embeddings/fastembed.py`
- `services/repo_semantic/embeddings/tei.py`

Перенести в:

- `C:\cursor_mcp\repo-semantic-search\services\repo_semantic\embeddings\*`

### 7.5 Deploy

- `deploy/repo-semantic-search/.env.example`
  -> `C:\cursor_mcp\repo-semantic-search\deploy\repo-semantic-search\.env.example`
- `deploy/repo-semantic-search/docker-compose.repo-semantic-search.yml`
  -> `...\deploy\repo-semantic-search\docker-compose.repo-semantic-search.yml`
- `deploy/repo-semantic-search/README.md`
  -> `...\deploy\repo-semantic-search\README.md`

### 7.6 Helper scripts

- `scripts/agents/ensure_repo_semantic_search.ps1`
  -> `C:\cursor_mcp\repo-semantic-search\scripts\agents\ensure_repo_semantic_search.ps1`
- `scripts/agents/register_repo_semantic_search.ps1`
  -> `C:\cursor_mcp\repo-semantic-search\scripts\agents\register_repo_semantic_search.ps1`

### 7.7 Docs

- `docs/specifications/platform/P1.Z-repo-semantic-search-mcp-specification.md`
  -> `C:\cursor_mcp\repo-semantic-search\docs\specifications\P1.Z-repo-semantic-search-mcp-specification.md`
- `docs/specifications/platform/REPO-SEMANTIC-SEARCH-MIGRATION-MAP.md`
  -> `C:\cursor_mcp\repo-semantic-search\docs\specifications\REPO-SEMANTIC-SEARCH-MIGRATION-MAP.md`

---

## 8) Что не копировать физически

Не копировать:
- `__pycache__/`
- `.pyc`
- любые docker volumes
- user-specific runtime secrets
- временные backup-файлы от локальных config scripts

---

## 9) Что нужно поправить сразу после переноса

### 9.1 Repo root defaults

Проверить:
- `services/repo_semantic/config.py`

Сейчас `_default_repo_root()` рассчитывается от текущего layout.
Если layout сохранится как в safe landing, оно останется рабочим.

### 9.2 Dockerfile / PYTHONPATH

Проверить:
- `apps/repo-semantic-mcp/Dockerfile`
- `deploy/repo-semantic-search/docker-compose.repo-semantic-search.yml`

Если layout не меняется, путь должен остаться рабочим.

### 9.3 Collection prefix

Сейчас дефолтный prefix всё ещё ориентирован на старое имя.

После переноса проверить и, скорее всего, переименовать:
- `SEMANTIC_MCP_COLLECTION_PREFIX`

Цель:
- убрать `vpn_server_repo` из generic tool.

### 9.4 Docs / wording

В новом проекте заменить формулировки вида:
- `source private monorepo`
- `repo-owned component внутри source monorepo`

на vendor-neutral описания.

---

## 10) Минимальная проверка после ручного переноса

После того как ты перенесёшь файлы, в новом проекте нужно прогнать:

### 10.1 Синтаксис

```powershell
py -3.12 -m compileall apps/repo-semantic-mcp services/repo_semantic
```

### 10.2 Подъём контура

```powershell
cd deploy/repo-semantic-search
docker compose -f docker-compose.repo-semantic-search.yml up -d --build
```

### 10.3 Готовность MCP

```powershell
powershell -ExecutionPolicy Bypass -File scripts/agents/ensure_repo_semantic_search.ps1 -Build
```

### 10.4 Регистрация в локальных клиентах

```powershell
powershell -ExecutionPolicy Bypass -File scripts/agents/register_repo_semantic_search.ps1
```

### 10.5 Smoke через Python MCP client

Проверить:
- `initialize`
- `list_tools`
- `index_status`
- `semantic_search_code`
- `semantic_search_docs`

### 10.6 Reboot-like check

Минимум:
- `docker restart repo-semantic-mcp`
- затем снова `ensure_repo_semantic_search.ps1`

Цель:
- убедиться, что после первой сборки persistent index переиспользуется и cold-start не повторяется как full rebuild.

---

## 11) Acceptance criteria миграции

Миграция считается успешной, когда в новой папке:

1. semantic MCP стартует без зависимости от source monorepo;
2. `index_status` показывает ненулевые `code/docs` counts;
3. `semantic_search_code` и `semantic_search_docs` возвращают релевантные результаты;
4. helper scripts работают;
5. Codex и Claude могут подключиться к новому HTTP MCP;
6. source monorepo больше не является runtime dependency для нового проекта.

---

## 12) План на следующий этап после переноса

После успешного safe landing:

1. cleanup layout
- решить, остаёмся ли на `apps/services`, или переходим на `src/semantic_mcp`

2. убрать source-monorepo-специфику
- collection prefix
- include/exclude defaults
- wording в docs

3. ввести repo profiles
- `generic`
- `python-monorepo`
- `docs-heavy`
- позже, возможно, source monorepo

4. сделать GPU profile
- отдельный TEI GPU deploy profile
- env/doc contract
- latency/throughput measurements

5. подготовить GitHub публикацию
- root README
- quickstart
- config examples для Codex/Claude
- license
- `.gitignore`
- release checklist

---

## 13) Решение по стратегии

Рекомендуемая стратегия:

- **сейчас**: ручной safe landing в `C:\cursor_mcp\repo-semantic-search`
- **после первого green startup**: cleanup и generalization уже в новой репе
- **после cleanup**: GPU profile + GitHub publish

Это самый прагматичный путь: он минимизирует риск потерять уже рабочий runtime и не смешивает миграцию с большим архитектурным рефакторингом.
