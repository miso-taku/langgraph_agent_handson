# LangGraph Human-in-the-Loop エージェントデモ

LangGraphのチェックポイント、Human-in-the-Loop、ストリーミング機能を活用したAIエージェントのデモアプリケーションです。
ユーザーからのリクエストを調査し、Web検索結果をHTMLレポートとして保存します。

## 特徴

- **Human-in-the-Loop**: ツール実行前にユーザー承認を要求
- **ストリーミング処理**: リアルタイムでエージェントの思考プロセスを表示
- **チェックポイント**: 会話状態を保存し、セッション間で継続可能
- **設計原則準拠**: カプセル化、単一責任、関心の分離を徹底した実装

## アーキテクチャ

### agent_core.py - エージェントコア

オブジェクト指向設計原則に基づいた、責務分離されたクラス構成:

#### 値オブジェクト
- **ToolApprovalRequest**: ツール承認データの不変オブジェクト

#### ドメインクラス
- **ToolRegistry**: ツール管理（Web検索、ファイル書き込み）
- **ToolApprovalRequestFactory**: 承認リクエスト生成
- **ToolExecutionApprover**: Human-in-the-Loop承認制御
- **AgentLLMInvoker**: Claude 3.7 Sonnet呼び出し管理
- **ToolExecutor**: ツール並列実行
- **ResearchAgent**: エージェント全体のオーケストレーション

#### タスク関数
- **_invoke_llm_task**: LLM呼び出しの非同期タスク
- **_execute_tool_task**: ツール実行の非同期タスク

### st_app.py - Streamlit UI

UIの責務を明確に分離したクラス構成:

#### 値オブジェクト
- **SessionState**: セッション状態の不変データモデル

#### UIコンポーネント
- **SessionStateManager**: Streamlitセッション状態管理
- **AgentStreamProcessor**: エージェント実行のストリーミング処理
- **MessageDisplayRenderer**: チャットメッセージ表示
- **UserFeedbackCollector**: ツール承認/拒否フィードバック収集
- **ToolApprovalRenderer**: ツール承認UI表示
- **ResearchAgentUI**: UI全体のオーケストレーション

## システムフロー

```
ユーザー入力
  ↓
ResearchAgentUI (UI統括)
  ↓
AgentStreamProcessor (ストリーミング処理)
  ↓
ResearchAgent (エージェント統括)
  ↓
AgentLLMInvoker (Claude呼び出し)
  ↓
ToolExecutionApprover (承認要求) → interrupt() → ユーザー承認待ち
  ↓
ToolExecutor (並列実行)
  ├─ TavilySearch (Web検索)
  └─ FileManagementToolkit (レポート保存)
  ↓
最終結果表示
```

## 環境構築

### 必要な環境変数

`.env`ファイルを作成し、以下を設定:

```bash
# AWS Bedrock認証情報
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# Tavily API Key (Web検索用)
TAVILY_API_KEY=your_tavily_api_key
```

### インストール

```bash
# uvを使用する場合
uv sync

# pipを使用する場合
pip install -r requirements.txt
```

### 依存パッケージ

- `langgraph`: エージェントフレームワーク
- `langchain`: LLM統合
- `langchain-community`: コミュニティツール
- `langchain-aws`: AWS Bedrock統合
- `langchain-tavily`: Web検索
- `streamlit`: Webアプリケーションフレームワーク
- `python-dotenv`: 環境変数管理

## 実行方法

```bash
# Streamlitアプリを起動
uv run streamlit run st_app.py

# または
streamlit run st_app.py
```

ブラウザで `http://localhost:8501` にアクセス

## 使い方

1. **リクエスト入力**: チャット入力欄に調査したいトピックを入力
   ```
   例: 「2025年のAI技術トレンドを調査してください」
   ```

2. **Web検索承認**:
   - エージェントがWeb検索を提案
   - 検索キーワードが表示される
   - 「承認」または「拒否」を選択

3. **レポート保存承認**:
   - 調査結果をHTML形式でプレビュー表示
   - 保存先ファイル名が表示される
   - 「承認」または「拒否」を選択

4. **結果確認**:
   - 承認後、`report/`ディレクトリにHTMLファイルが保存される
   - 最終結果がUI上に表示される

## 設計原則

本プロジェクトは `design_principles.md` に記載された以下の原則に準拠:

### カプセル化
- クラスが単体で正常動作
- インスタンス変数への安全なアクセス
- 値オブジェクトによる不変性の保証

### 単一責任原則
- 各クラスが1つの責務のみを担当
- 目的が異なる場合は共通化せず分離

### 関心の分離
- インターフェースと実装を分離
- 依存方向は上位→下位のみ

### 目的駆動命名
- `ToolExecutionApprover`: 存在(Approver)ではなく目的(ツール実行承認)
- `ResearchAgent`: 抽象的な"Agent"ではなく具体的な"調査"エージェント
- `SessionStateManager`: 状態の"管理"という目的を明示

### Googleスタイルdocstring
- 全クラス・メソッドに目的と仕様を記載
- Args/Returns/Raisesを明確に記述

## ディレクトリ構造

```
langgraph_agent_handson/
├── agent_core.py           # エージェントコアロジック
├── st_app.py               # Streamlit UIアプリケーション
├── design_principles.md    # 設計原則ドキュメント
├── README.md               # 本ファイル
├── .env                    # 環境変数（要作成）
├── pyproject.toml          # プロジェクト設定
└── report/                 # 出力レポート保存先（自動生成）
```

## 技術スタック

- **LLM**: Claude 3.7 Sonnet (AWS Bedrock)
- **エージェントフレームワーク**: LangGraph
- **Web検索**: Tavily Search API
- **UI**: Streamlit
- **非同期処理**: LangGraph Task API
- **状態管理**: LangGraph MemorySaver

## トラブルシューティング

### AWS Bedrock接続エラー
```
botocore.exceptions.NoCredentialsError
```
→ `.env`にAWS認証情報が正しく設定されているか確認

### Tavily API エラー
```
langchain_tavily.exceptions.TavilyAPIError
```
→ Tavily APIキーが有効か確認

### モジュール未インストールエラー
```
ModuleNotFoundError: No module named 'xxx'
```
→ `uv sync` または `pip install -r requirements.txt` を実行

## ライセンス

本プロジェクトのライセンスについては `LICENSE` ファイルを参照してください。

## 参考リンク

- [LangGraph公式ドキュメント](https://langchain-ai.github.io/langgraph/)
- [Streamlit公式ドキュメント](https://docs.streamlit.io/)
- [AWS Bedrock](https://aws.amazon.com/bedrock/)
- [Tavily Search API](https://tavily.com/)
