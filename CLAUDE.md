# CLAUDE.md

@AGENTS.md

## このファイルの役割

Claude Code が読み込むのは `CLAUDE.md` だけで、`AGENTS.md` は読み込みません。冒頭の import で `AGENTS.md` を取り込むことで、他のコーディングエージェントと同じルールを共有します。

プロジェクト固有の指示は `AGENTS.md` 側に書きます。このテンプレートを使って `CLAUDE.md` を更新する場合は、Claude Code 固有の指示を対象プロジェクト側で手動統合してください。Claude Code 固有の記載が不要な場合は、`CLAUDE.md` を `AGENTS.md` へのシンボリックリンクにしても同じ結果になります（Windows では管理者権限または開発者モードが必要なため、import 方式を推奨します）。

## タスク別コマンド

`.claude/commands/` に必要なコマンドを配置した場合、次のコマンドを使用できます。引数には対象の URL または番号を渡します。

| コマンド | 用途 |
| --- | --- |
| `/implement-issue` | Issue の調査、最小差分の実装、検証、報告 |
| `/fix-ci` | CI 失敗の原因特定、最小修正、再検証 |
| `/review-pr` | 品質・セキュリティ・互換性のレビュー（コードは変更しない） |
| `/investigate-issue` | コードを変更しない原因調査と対応案の比較 |
| `/audit-repository` | リポジトリの課題・改善点を診断し、必要に応じて重複のないGitHub Issueを作成 |
| `/propose-features` | アプリの目的・実装・拡張性を確認し、追加実装候補を必要に応じてGitHub Issueとして作成 |

## 調査・計画レビュー・実装・差分レビューの段階分けとサブエージェントへの委任

`/implement-issue`、`/fix-ci`、`/improve-tests`、`/refactor-repository`、`/update-dependencies`、および `/quick-request` の `実装` / `CI` / `テスト` / `リファクタ` / `依存更新` は、次の4段階を、プロンプト本文（ツール中立）に既に含んでいます。ChatGPT Work を含むどのツールでも、この段階分けと2つの必須ゲートは自己レビューとして機能します。

1. 調査（目的・完了条件・変更範囲・実装方針・リスクを整理する）
2. **計画レビュー**（要求との整合性・見落とし・実現性を確認する。Critical/High 相当の懸念があれば着手前に解消する必須ゲート）
3. 実装
4. **差分レビュー**（`review-pr.md` と同じ観点で確認する。Critical/High 相当の指摘があれば完了・PR作成前に解消する必須ゲート）

`agents/` を `.claude/agents/` に配置している場合、Claude Code ではこの4段階を独立したコンテキストの `ai-platform-planner`（調査）/ `ai-platform-reviewer`（計画レビュー・差分レビューの両方）/ `ai-platform-implementer`（実装）サブエージェントに委任してください。同じ会話の中で自分がすべての段階をこなすより、権限分離（planner と reviewer は Write/Edit を持たない）と独立したコンテキストによる客観性が得られます。委任する場合の手順は次のとおりです。

1. `ai-platform-planner` に、対象の Issue/PR/CI失敗ログと完了条件を渡して調査を委任し、変更範囲・実装方針・リスクを整理させる。
2. その計画を `ai-platform-reviewer` に渡して計画レビューを委任する。Critical または High の指摘がある場合、実装に進まず、`ai-platform-planner` に計画を練り直させてから再度レビューする。
3. 計画レビューで Critical/High の指摘がなくなってから実装する（`ai-platform-implementer` に委任するか、自身で実装する）。
4. 実装後、差分とテスト結果を `ai-platform-reviewer` に渡して差分レビューを委任する。
5. 差分レビューで Critical または High の指摘がある場合、PR を作成せず、指摘に対応してから再度レビューする。Medium 以下の指摘は、対応するか対応しない理由を報告に残す。
6. 差分レビューで Critical/High の指摘がなくなってから PR を作成する。

`.claude/agents/` を配置していない、またはコードを変更しない `/review-pr`、`/investigate-issue`、`/audit-repository`、`/propose-features`、`/security-review` にはサブエージェントへの委任もこの段階分けの拡張も不要です（後者はプロンプト本文の指示どおり単独で完結します）。

<!-- サブエージェントへの委任は Claude Code 固有の強化です。この節は AGENTS.md 側には反映しません。ベースとなるレビューゲートは prompts/ 本文にあるため、この節を適用しなくても機能します。 -->

## 検証コマンドの実行

`AGENTS.md` の「検証コマンド」を実行する前に、依存関係が導入済みかを確認します。クラウドセッションは毎回新しい VM でリポジトリを clone するため、ローカルにだけ導入した依存やツールは存在しません。導入が必要な場合は、未導入であることを報告に含めます。

<!-- 依存導入を自動化する場合は、リポジトリの .claude/settings.json に SessionStart hook を設定します。設定例は AI Platform Repository の README を参照してください。 -->

## 報告とコミット

- 実行していない検証を成功として報告しません。実行できなかった検証は、理由とともに明示します。
- コミットとプッシュは、依頼された場合にのみ行います。クラウドセッションからプッシュできるのは、そのセッションの作業ブランチだけです。
- PR を作成する場合は、`.github/pull_request_template.md` の項目を埋めます。

<!--
このテンプレートを `CLAUDE.md` に適用する場合は、ファイル全体が置き換わります。
プロジェクト固有の指示は AGENTS.md に記載してください。
Claude Code 固有の指示（plan mode を使う範囲、レビューが必須のディレクトリ、
優先して使うサブエージェントなど）は、適用前に対象プロジェクト側で手動統合してください。
-->
