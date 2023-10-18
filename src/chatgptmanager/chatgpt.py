import os
from pathlib import Path
import pickle
from collections.abc import Iterable

import openai
import numpy as np
import pandas as pd

class Chat:
    def __init__(self, api_key, model_name="gpt-3.5-turbo", embedding_model_name="text-embedding-ada-002",
                 interactive=True, auto_saveload=True):
        """
        Chat クラスの初期化.

        Parameters
        ----------
        api_key : str
            OpenAI APIのキー.
        model_name : str, optional
            使用するOpenAIモデルの名前. Default is "gpt-3.5-turbo".
        embedding_model_name : str, optional
            使用する埋め込みモデルの名前. Default is "text-embedding-ada-002".
        interactive : bool, optional
            連続した会話を行うか(過去の会話の履歴を持つか). Default is True.
        """
        self.api_key = openai.api_key = api_key
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        
        # トークンあたりの価格データ
        self.pricing_data = {
            "gpt-3.5-turbo": {
                "input": 1.5e-6,
                "output": 2.0e-6,
            },
            "gpt-4": {
                "input": 3.0e-5,
                "output": 6.0e-5,
            },
            "text-embedding-ada-002": {
                "input": 1.0e-7,
            }
        }
        
        # 実際に使った分の料金
        self.fee = 0
        
        # 返答のキャッシュ
        self.cache = {}
        
        self.savedir = Path("./chatgpt/")
        self.savedir.mkdir(exist_ok=True)
        
        # 会話履歴
        self.chat_history = []
        
        # 連続した会話を行うか(過去の会話の履歴を持つか)
        self.interactive = interactive
        
        self.auto_saveload = auto_saveload
        if self.auto_saveload:
            self.load()
        
    def reset(self):
        """
        会話履歴をリセットする.
        """
        self.chat_history = []
        
    def save(self):
        """
        キャッシュを保存する.
        """
        path = self.savedir / f"{pd.Timestamp('now').strftime('%Y%m%d%H%M%S')}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.cache, f)
            
    def load(self, path=None):
        """
        キャッシュをロードする.

        Parameters
        ----------
        path : str, optional
            キャッシュファイルのパス. Default is None.
        """
        if path is None:
            for path in self.savedir.glob('*.pkl'):
                self.load(path)
        else:
            with open(path, "rb") as f:
                obj = pickle.load(f)
                assert isinstance(obj, dict)
                self.cache |= obj

    def to_str(self):
        """
        会話履歴をstringに変換して返す.

        Returns
        -------
        str
            会話履歴の文字列.
        """
        speech_strs = [f"> {speech['role']}: \n{speech['content']}" for speech in self.chat_history]
        return '\n\n'.join(speech_strs)
                
    def calculate_price(self, input_tokens, output_tokens):
        """
        現在のモデルでの利用料金(ドル)を返す.
        
        Parameters
        ----------
        input_tokens : int
            プロンプトとして入力する文字数.
        output_tokens : int
            出力の文字数.
            
        Returns
        -------
        price : float
            利用料金(ドル).
        """
        pricing = self.pricing_data[self.model_name]
        return pricing["input"] * input_tokens + pricing["output"] * output_tokens
        
    def estimate_price(self, message):
        """
        messageを送った際の料金(ドル)の目安を返す.

        Parameters
        ----------
        message : str
            送信されるメッセージ.

        Returns
        -------
        float
            予想される利用料金(ドル).
        """
        return self.calculate_price(
            input_tokens = len(message),
            output_tokens = len(message) * 1.1 + 20,
        )
    
    def summarize_and_clear_history(self):
        # あまり会話が長くなると1回あたりの料金が高くなるため、会話履歴を要約する
        self('今までの会話を要約してください。')
        
        # 会話履歴を「今までの会話を要約してください」「(ChatGPTの返答)」のみにする
        self.chat_history = self.chat_history[-2:]
        
    def __call__(self, message, temperature=0, temporary_interactive=False):
        """
        モデルを使用して会話を行う.

        Parameters
        ----------
        message : str
            ユーザーからのメッセージ.
        temperature : float, optional
            モデルの出力のランダム性を制御する. Default is 0.
        temporary_interactive : bool, optional
            一時的に連続した会話を行うか. Default is False.

        Returns
        -------
        str
            モデルの返答.
        float
            利用料金(ドル).
        """
        message = str(message)
        if temporary_interactive:
            saved_param = {"interactive": self.interactive}
            self.interactive = True
            
        if not self.interactive:
            # 連続して会話を行わない場合 会話の履歴を消去する
            self.chat_history = []
        
        if (not self.chat_history) and (temperature == 0):
            # messageが同じなら同じ返答をする設定の場合
            key = (self.model_name, message)
            if key in self.cache:
                res = self.cache[key]
                self.chat_history.append({'role': 'user', 'content': message})
                self.chat_history.append({'role': 'assistant', 'content': res})
                price = 0
                return res, price
            
        # チャットで呼び出す
        my_message = {'role': 'user', 'content': message}
        
        completion = openai.ChatCompletion.create(
          model = self.model_name,
          messages = self.chat_history + [my_message],
          temperature = temperature,
        )
        
        # 返答を取得
        res = completion["choices"][0]["message"]["content"]
        self.chat_history.append(my_message) # atomicな操作にするため、送信&返答が終わってから追加する
        self.chat_history.append({'role': 'assistant', 'content': res})
        
        if (len(self.chat_history) == 2) and (temperature == 0):
            # 一問一答形式かつmessageが同じなら同じ返答をする設定の場合
            # 返答を保存する
            self.cache[(self.model_name, message)] = res
        
        # 料金を計算
        price = self.calculate_price(
            input_tokens = completion["usage"]["prompt_tokens"],
            output_tokens = completion["usage"]["completion_tokens"],
        )
        self.fee += price
        
        if temporary_interactive:
            self.interactive = saved_param["interactive"]
            
        if self.auto_saveload:
            self.save()
            
        return res, price
    
    def embedding(self, query):
        """
        指定されたクエリの埋め込みベクトルを取得する.

        Parameters
        ----------
        query : str or Iterable
            埋め込みベクトルを取得するためのクエリ.

        Returns
        -------
        np.array or tuple
            クエリの埋め込みベクトル. クエリが複数の場合は、埋め込みベクトルの配列と料金がタプルで返される.
        """
        if (not isinstance(query, str)) and isinstance(query, Iterable):
            # 配列が入力されたとき　まとめて送る
            multiple_query = True
            inputs = [str(item).replace("\n", " ") for item in query]
        else:
            multiple_query = False
            inputs = [str(query).replace("\n", " ")]
            
        res = openai.Embedding.create(
            input = inputs, 
            model = self.embedding_model_name
        )
        price = res["usage"]["total_tokens"] * self.pricing_data[self.embedding_model_name]["input"]
        vecs = np.array([data["embedding"] for data in res["data"]])
        self.fee += price
        
        if multiple_query:
            return vecs, price
        else:
            return vecs[0], price
        
        
        