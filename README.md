# ChatGPTManager
OpenAIのAPIを使用して簡単にテキストベースの会話を行うためのラッパーモジュールです。
 
 <p align="center">
 <img src="https://img.shields.io/badge/python-v3.9+-blue.svg">
 <img src="https://img.shields.io/badge/contributions-welcome-orange.svg">
 <a href="https://opensource.org/licenses/MIT">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg">
 </a>
</p>

## Last Stable Release
```bash
pip install git+https://github.com/FumiYoshida/chatgptmanager
```
## Features

- OpenAIモデルと埋め込みモデルの両方をサポート
- キャッシュによる返答の保存
- トークンあたりの価格データ
- 実際に使用した分の料金の計算

## Usage

```python
from chatgptmanager import Chat

chat = Chat(api_key="YOUR_OPENAI_API_KEY")
response, price = chat("Hello, how are you?")
print(response)  # モデルの返答を表示
print(price)     # その返答の料金を表示
```

## Methods

### `__init__(api_key, model_name="gpt-3.5-turbo", embedding_model_name="text-embedding-ada-002", interactive=True)`

クラスの初期化メソッド。

### `reset()`

会話履歴をリセットします。

### `save()`

キャッシュを保存します。

### `load(path=None)`

キャッシュをロードします。

### `to_str()`

会話履歴を文字列として返します。

### `calculate_price(input_tokens, output_tokens)`

与えられたトークン数に基づいて、現在のモデルの利用料金を計算します。

### `estimate_price(message)`

メッセージを送信したときの料金の見積もりを返します。

### `__call__(message, temperature=0, temporary_interactive=False)`

モデルを使用して会話を行います。

### `embedding(query)`

指定されたクエリの埋め込みベクトルを取得します。
