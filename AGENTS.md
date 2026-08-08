# AGENTS.md — VLM SOP Monitor

> このファイルにはプロジェクト固有の指示だけを記載します。共通の詳細ルールは `.claude/rules/ai-platform-common.md`（正本は AI Platform Repository の `prompts/coding-agent-typescript-python.md`）を参照してください。参照できない環境では「AI Platform 共通ルール」の要約を適用します。

## プロジェクト概要

- 目的: 製造・組立・保守作業の映像を、物体検出・姿勢推定・追跡・時系列ルール・VLMで解析し、SOP（標準作業手順）の進行と安全違反を記録するローカル優先アプリケーションの PoC。
- 対象環境: ローカル実行および Docker。既定の `mock` モードはモデル・GPU・外部APIなしで動作します。
- 位置づけ: PoC です。人物解析・日常動作判定は試験実装であり、安全上重要な判定を VLM 単独で確定させません。

## 使用技術

- 言語・バージョン: Python 3.11 以上（CI は 3.11 と 3.12 で実行）。TypeScript は使用しません。
- フレームワーク・主要ライブラリ: FastAPI、Uvicorn、Pydantic v2、PyYAML、NumPy（`<2`）、OpenCV（`opencv-python-headless`）、標準ライブラリの `sqlite3`。
- 任意依存: `[vision]` は Ultralytics / MediaPipe / ONNX Runtime、`[local-vlm]` は Transformers / PyTorch / Pillow。いずれも選択時にのみ import します。
- フロントエンド: ビルド工程のない素の HTML / CSS / JavaScript（`templates/index.html`、`static/`）。npm もバンドラーも導入しません。

## パッケージマネージャー

- Python: `pip`。依存定義は `pyproject.toml`、ロックファイルはありません。導入は `python -m pip install -e ".[dev]"`。
- 依存追加は最小限にとどめます。`app/` の中核処理は標準ライブラリと既存依存で完結させ、重い依存は任意依存（extras）として遅延 import します。
- ライセンス上の制約があるため、モデル重み・学習データを含む依存を追加する場合は README「ライセンス上の注意」に沿って可否を確認します。

## ディレクトリ構成

| パス | 役割 | 変更時の注意 |
| --- | --- | --- |
| `app/main.py` | FastAPI のエントリーポイント（`create_app()`）と全 HTTP / WebSocket 経路 | ルート定義順に依存します。`/api/events/download` は `/api/events/{event_id}` より前に置きます |
| `app/services/session.py` | セッション管理、フレーム処理、イベント記録、配信の統合 | 唯一の状態保持点です。処理順（イベント記録 → 証拠フレーム → 配信）を崩さないでください |
| `app/vision/` | 検出・姿勢・追跡・フレームソース・パイプライン | `source.py` はスレッドと asyncio の境界です。ロックとイベントの扱いを変更する場合は必ずテストを追加します |
| `app/sop/` | SOP 定義の読み込み、条件評価、状態機械 | 状態機械は前進のみです。工程の巻き戻しを導入しません |
| `app/vlm/` | VLM プロバイダーとスキーマ、プロンプト | 応答は必ず `parse_vlm_response` で検証します。自由文をそのまま信頼しません |
| `app/storage/` | SQLite 接続・リポジトリ・証拠フレーム保存 | SQL は必ずプレースホルダーを使用します。文字列連結で値を埋め込みません |
| `app/core/` | 設定、ロギング、メトリクス、認証・レート制限 | `config.py` に項目を追加したら `config/default.yaml`・`.env.example`・README を同時に更新します |
| `config/`, `sop/` | 既定設定と SOP 定義（YAML） | `config/default.yaml` は設定の正本です。新設定は既定値付きで追加します |
| `templates/`, `static/` | ダッシュボード | 挿入値は必ず `escapeHtml` を通します。依存追加やビルド導入はしません |
| `tests/` | 外部依存なしで完結する pytest | ネットワーク・実カメラ・実モデルに依存するテストを追加しません |
| `scripts/` | デモ生成、依存導入などの補助スクリプト | 標準ライブラリとプロジェクト依存のみを使用します |

## アーキテクチャ

- 処理の流れ: フレームソース → 検出・姿勢推定 → IoU 追跡 → `Observation` → SOP 条件評価・状態機械 → イベント記録（SQLite・証拠フレーム）→ WebSocket / SSE 配信。
- VLM は毎フレーム呼び出しません。間隔・検出内容の変化・手動実行のいずれかで起動し、バックグラウンドタスクで実行してフレーム処理を止めません。失敗時はバックオフします。
- フレームソースは `app/vision/source.py` に隔離します。`cv2.VideoCapture` はブロックするため専用スレッドで動作し、破棄先入れ（drop-oldest）の有界バッファ経由で asyncio 側へ渡します。解析が遅れた場合はレイテンシを蓄積せずフレームを捨てます。
- 重要な設計制約:
  - `app/storage` はビジョン・SOP のドメイン型に依存しません（保存対象は辞書と単純な属性参照に限定）。
  - モック実装（`MockDetector` / `MockPoseEstimator`）はフレーム ID に対して決定的です。テストがこのタイムラインに依存しているため、挙動を変える場合はテストも同時に更新します。
  - ブラウザ配信（`browser`、レガシー別名 `camera` / `video`）とサーバー側取り込み（`server_camera` / `file` / `rtsp`）は別系統です。別名の意味を変えると既存クライアントがサーバーのカメラを開こうとします。

## 検証コマンド

| 目的 | コマンド | 実行条件・補足 |
| --- | --- | --- |
| 依存導入 | `python -m pip install -e ".[dev]"` | 新しいセッションでは最初に必要です |
| lint | `make lint`（`python -m ruff check app tests`） | 必須。ルール選択は `pyproject.toml` にあります |
| 整形 | `make format`（`python -m ruff format app tests`） | 任意。既存の密な一行スタイルを大きく崩す整形は行いません |
| テスト | `make test`（`python -m pytest -q`） | 必須。外部依存なしで完結します |
| 型チェック | 未導入 | mypy 等は導入していません。型ヒントは付けますが、型チェックの実行を成功として報告しません |
| 手動起動 | `make dev` | `http://localhost:8000` でダッシュボードを確認します |
| デモ | `make demo` | 疑似動画を生成して 60 フレームを解析します |
| ビルド | 該当なし | パッケージ配布は行っていません |

## 変更禁止領域

- セキュリティ境界: `app/vision/source.py` の `resolve_source_uri`（`file` ソースを `./data` 配下に限定）、`app/storage/frames.py` の `FrameStore.path`（パストラバーサル検査）、`app/core/security.py` の `secrets.compare_digest` による定数時間比較。緩和が必要な場合は理由と代替策を PR に明記します。
- 秘匿情報の露出防止: `/api/config` は `vlm.api_key` と `security.api_key` をマスクします。設定値をそのまま返す変更を加えません。
- SQLite スキーマ: 列を追加する場合は `app/storage/database.py` の `_migrate()` に既存データベースの移行処理を必ず併せて追加します。破壊的なスキーマ変更は行いません。
- タイムスタンプ: 保存する時刻は `utc_now_iso()` による UTC 表記に統一します。ローカルオフセットで保存すると保持期間の文字列比較が壊れます。
- VLM プロンプト（`app/vlm/prompts.py`）と応答スキーマ: 出力の JSON Schema と対応関係があります。変更時は `tests/unit/test_vlm.py` と `tests/unit/test_openai_provider.py` を同時に確認します。
- ライセンス表記（README「ライセンス上の注意」）: モデル重み・学習データの利用条件に関する記述を、確認なく緩めません。

## DB・API 固有ルール

- DB: `sqlite:///` 形式のみ対応します。WAL モードで動作し、`-wal` / `-shm` の副生成物が同じディレクトリに作られます。全 SQL はプレースホルダー使用、`Repository` のロック保持を維持します。保持期間の削除は行と証拠フレームの両方を対象にします。
- API: 既存経路の互換性を維持します。`GET /api/events` 系は JSON 配列を返し、総件数は `X-Total-Count` ヘッダーで返します。ページングの既定値と上限（`limit` の上限 1000、`limit<=0` は CSV 出力用の全件）を変更する場合は README を更新します。
- 認証・レート制限は既定で無効です。既定を有効側へ変更しません（同梱ダッシュボードは API キーを送信しないため、既定を変えるとデモが動かなくなります）。
- 外部サービス: `openai_compatible` プロバイダーは映像・観測情報を外部へ送信し得ます。送信内容を増やす変更は、README のプライバシー記述と併せて検討します。

## デプロイ上の注意

- 秘密情報は環境変数で渡します。`.env` はコミットしません。`.env.example` には値を書かず、項目名と既定値・説明のみを記載します。
- Docker は既定で CPU の mock モードです。GPU を使う構成をこのリポジトリの既定にしません。
- 本番では認証・認可・TLS・レート制限・マルウェア検査を別途追加する前提です。README の記述と実装状態を一致させてください。
- 顔・人物映像を扱うため、保持期間（`RETENTION_DAYS`）と証拠フレーム保存（`SAVE_EVENT_FRAMES`）の既定を、運用ポリシーの確認なしに緩めません。

## プロジェクト固有の完了条件

- `make lint` と `make test` が成功していること。実行できなかった検証は理由とともに明記します。
- 設定項目を追加・変更した場合は、`app/core/config.py`・`config/default.yaml`・`.env.example`・README のすべてを更新していること。
- API を追加・変更した場合は README の API 表を更新し、統合テスト（`tests/integration/`）を追加していること。
- 不具合修正には回帰テストを追加していること。モックのタイムラインに依存する変更では、依存するテストを併せて更新していること。
- UI を変更した場合は、`templates/index.html` の要素 ID と `static/app.js` の参照が一致していることを確認していること。

## AI Platform 設定

このリポジトリには `sj55576/ai-platform` の共通設定を適用しています。

| 配置先 | 内容 |
| --- | --- |
| `CLAUDE.md` | `AGENTS.md` を取り込む Claude Code 用ブリッジ |
| `.claude/rules/ai-platform-common.md` | 共通ルールの全文 |
| `.claude/commands/` | タスク別プロンプト（`/implement-issue`、`/fix-ci` ほか） |
| `.github/pull_request_template.md`、`.github/ISSUE_TEMPLATE/ai-platform.yml` | PR / Issue テンプレート |

適用元リビジョン: `4c2216b081f3900ac2dc2aadacf146f228a22e9d`。更新は `/sync-ai-platform sj55576/VLMPoC PR作成` で差分を確認します。下記マーカー区間だけが同期対象で、それ以外のプロジェクト固有の記述は上書きしません。

<!-- AI-PLATFORM:START -->
## AI Platform 共通ルール（同期管理）

- 変更前に関連実装・設定・テストを確認し、既存の設計と命名を尊重する。
- 必要最小限の差分を選び、外部入力を検証する。TypeScript は型安全性、Python は型ヒントと明確な例外処理を優先する。
- 不具合修正には回帰テストを追加し、lint・型チェック・テスト・ビルドを実行する。テスト削除やチェック無効化で問題を回避しない。
- Secret・個人情報を出力しない。認証、認可、DB、公開 API は根拠なく変更しない。
- 実行していない検証を成功と報告せず、PR には変更内容、テスト結果、リスク・未検証事項を記載する。

詳細: `sj55576/ai-platform` の `prompts/coding-agent-typescript-python.md`
<!-- AI-PLATFORM:END -->
