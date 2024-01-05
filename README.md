# Faiss Text Search

## 概要

OpenAIの埋め込みとFaiss（Facebook AI Similarity Search）を組み合わせて、大量のテキストデータから最も類似したテキストを高速に検索します。

## 必要な環境

- Python 3.11
- Poetry

## インストール

1. このリポジトリをクローンします：
```shell
git clone https://github.com/yourusername/faiss-text-search.git
cd faiss-text-search
```

2. Poetryを使用して依存関係をインストールします：

```shell
poetry install
```


## 使用方法

1. メインのPythonスクリプトを実行します：

```shell
poetry run python main.py
```

このスクリプトは、`data/sample.csv`からテキストデータを読み込み、それらのテキストをOpenAIで埋め込み、Faissを使用して類似度検索を行います。結果は標準出力に表示されます。

## 注意事項

このリポジトリの実行にはOpenAIのAPIキーが必要です。

参照：https://note.com/komzweb/n/n3392c290d7b8
