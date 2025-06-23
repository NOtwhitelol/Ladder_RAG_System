def STANDALONE_QUESTION_REWRITE_TEMPLATE(chat_history, follow_up_question):
    context = "\n".join([f"**{item['role']}:**\n```\n{item['content']}\n```\n" for item in chat_history])

    return [{"role": "system", "content":
f"""Given a chat history and a follow up question, which might reference context in the chat history.
Help user formulate a standalone question, which can be understood without the chat history.
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.

Your response will be a JSON object with the following format:
{{"question": "rewritten standalone question"}}

---
Here are some examples:

### Example 1:
- CHAT HISTORY: 
**user:**
```
Hi, I'm learning machine learning. Can you explain overfitting?
```

**assistant:**
```
Sure! Overfitting happens when a model learns noise instead of patterns, performing well on training data but poorly on new data.
```

**user:**
```
How can I prevent overfitting in my model?
```

**assistant:**
```
You can use techniques like regularization (L1/L2), dropout, early stopping, and increasing training data to reduce overfitting.
```

**user:**
```
What is dropout, and how does it help?
```

**assistant:**
```
Dropout randomly deactivates some neurons during training, forcing the network to learn more general patterns and preventing over-reliance on specific neurons.
```

**user:**
```
What's the difference between L1 and L2 regularization?
```

**assistant:**
```
L1 adds absolute weight values as a penalty, leading to sparse weights. L2 adds squared weight values, reducing large weights but keeping all features.
```

**user:**
```
When should I use L1 versus L2 regularization?
```

**assistant:**
```
Use L1 if you want feature selection since it removes less important features. Use L2 for smoother weight distributions and better generalization.
```

- FOLLOW UP USER QUESTION: 
```
Can I use both of them?
```

- STANDALONE QUESTION: 
```
{{"question": "Can I use both L1 and L2 regularization?"}}
```


### Example 2:
- CHAT HISTORY: 
**user:**
```
affine transform
```

**assistant:** 
```
An affine transform is a linear mapping method that preserves points, straight lines, and planes in geometry and computer graphics. 
It includes operations like translation, scaling, rotation, reflection, and shearing. 
Affine transformations maintain the "affine properties" of a shape, meaning parallel lines remain parallel, and midpoints of line segments are preserved, but angles and lengths may change.
Affine transformations are widely used in image processing, computer vision, and machine learning applications such as data augmentation in training datasets for better model generalization.
```

- FOLLOW UP QUESTION: 
```
is it in-plane rotation or out-of-plane rotation
```

- STANDALONE QUESTION: 
```
{{"question": "Is affine transform in-plane rotation or out-of-plane rotation?"}}
```


### Example 3:
- CHAT HISTORY: 
**user:**
```
RNN的發展歷史
```

**assistant:**
```
循環神經網路（Recurrent Neural Network, RNN）的發展歷史可以追溯到20世紀80年代，隨著深度學習和計算資源的進步，其架構和應用逐步演進。以下是RNN發展的主要里程碑：
1. RNN的起源與早期發展
1980年代初期：RNN的概念由John Hopfield提出，他提出了Hopfield網絡作為一種具有迴圈結構的神經網絡，用於存儲記憶。
1986年：David Rumelhart等人提出反向傳播算法（Backpropagation Through Time, BPTT），為RNN的訓練奠定了基礎。BPTT能有效處理時間序列數據中的時間依賴關係。
2. 梯度消失與爆炸問題的挑戰（1990年代）
RNN在長序列數據上的訓練遭遇了梯度消失和梯度爆炸的問題，使得它難以捕捉長期依賴（Long-term Dependency）。
研究者逐步發現，標準RNN在處理長序列的記憶能力有限，這限制了其應用範圍。
3. LSTM的誕生（1997年）
1997年：Hochreiter和Schmidhuber提出了長短期記憶網絡（Long Short-Term Memory, LSTM），通過引入門控機制（如輸入門、遺忘門和輸出門）來解決梯度消失問題。
LSTM的設計使得網絡能夠學習和記住長期依賴資訊，成為時間序列建模的重要工具。
```

- FOLLOW UP QUESTION: 
```
上網搜尋2025年RNN相關論文
```

- STANDALONE QUESTION: 
```
{{"question": "上網搜尋2025年RNN相關論文"}}
```


### Example 4:
- CHAT HISTORY: 
**user:**
```
hi
```

**assistant:**
```
Welcome to Ladder! I'm here to help you with any questions or issues you may have while building your artificial neural network models. 

What would you like to do today? Would you like to:

Create a new model?
Train an existing model?
Evaluate the performance of a model?
Get started with a tutorial or guide?
Let me know, and I'll be happy to assist you!
```

- FOLLOW UP USER QUESTION: 
```
start from the beginning
```

- STANDALONE QUESTION: 
```
{{"question": "Help me build an artificial neural network model on Ladder from the beginning."}}
```

### Example 5:
- CHAT HISTORY: 
**user:**
```
我訓練了一個模型，但它的預測結果偏向多數類別。我該如何解決這個問題？
```

**assistant**
```
這可能是類別不平衡的問題。你可以嘗試重新取樣（對少數類別進行過採樣或對多數類別進行欠採樣），或使用類別加權的損失函數。
```

**user:**
```
我試過過採樣，但我的模型仍然偏向多數類別。還有其他方法嗎？
```

**assistant:**
```
你可以使用合成數據生成方法，例如 SMOTE,或將評估指標改為 F1-score, 而不是準確率, 以獲得更好的評估結果。
```

- FOLLOW UP USER QUESTION: 
```
我用了 SMOTE, 但模型在真實數據上的表現變差了。有更好的方式來生成合成樣本嗎?
```

- STANDALONE QUESTION: 
```
{{"question": "我用了SMOTE來避免偏向多數類別, 但模型在真實數據上的表現變差了。有更好的方式來生成合成樣本嗎?"}}
```

### Example 6:
- CHAT HISTORY: 
**user:**
```
how to open the result folder
```

**assistant**
```
According to the "Start using Ladder" content, you can access the results folder by selecting the option "Open a Results Folder" from the starting screen of Ladder. This will allow you to view the training logs based on your user settings.

Alternatively, after executing the Python file generated through NOM and the program finishes running, you can upload the results folder back into Ladder to view the training logs.
```

- FOLLOW UP USER QUESTION: 
```
翻譯成中文
```

- STANDALONE QUESTION: 
```
{{"question": "翻譯成中文"}}
```

### Example 7:
- CHAT HISTORY: 
**user:**
```
怎麼訓練我的模型
```

**assistant**
```
你可以在Ladder的右側工具箱中調整訓練參數和超參數，包括epoch數、交叉驗證類型、追蹤頻率、批次大小、資料洗牌、記錄頻率、儲存頻率、測試頻率、權重頻率、跟蹤計數和跟蹤頻率等。當你完成模型編輯後，可以點擊右上角的「Project」按鈕，將專案儲存為NOM格式（project.json）或Python腳本（train.py）。如果你想立即訓練模型，可以點擊右下角的「Train Now」按鈕，按照指示進行。
```

**user:**
```
我訓練完模型了 該怎麼查看結果
```

**assistant:**
```
你可以點擊右下角的「Open a results folder」按鈕，將你的訓練結果上傳到Ladder中。然後，你就可以在Ladder中檢視你的訓練記錄和追蹤記錄。

```

- FOLLOW UP USER QUESTION: 
```
最近有甚麼機器學習相關的新聞?
```

- STANDALONE QUESTION: 
```
{{"question": "近期機器學習相關新聞"}}
```

### Example 8:
- CHAT HISTORY: 
**user:**
```
怎麼訓練我的模型
```

**assistant**
```
你可以在Ladder的右側工具箱中調整訓練參數和超參數，包括epoch數、交叉驗證類型、追蹤頻率、批次大小、資料洗牌、記錄頻率、儲存頻率、測試頻率、權重頻率、跟蹤計數和跟蹤頻率等。當你完成模型編輯後，可以點擊右上角的「Project」按鈕，將專案儲存為NOM格式（project.json）或Python腳本（train.py）。如果你想立即訓練模型，可以點擊右下角的「Train Now」按鈕，按照指示進行。
```

**user:**
```
我訓練完模型了 該怎麼查看結果
```

**assistant:**
```
你可以點擊右下角的「Open a results folder」按鈕，將你的訓練結果上傳到Ladder中。然後，你就可以在Ladder中檢視你的訓練記錄和追蹤記錄。

```

- FOLLOW UP USER QUESTION: 
```
上網搜尋CNN相關論文
```

- STANDALONE QUESTION: 
```
{{"question": "上網搜尋CNN相關論文"}}
```

---
Remember: Your task is to help user formulate a standalone question, which can be understood without the chat history.
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
Respond with the result only. No justification or explanation.
Your response will be only a JSON object with the following format:
{{"question": "rewritten standalone question"}}
"""},

{"role": "user", "content":
f""" Formulate a standalone question, which can be understood without the chat history. DO NOT provide reason or other explanations.
- CHAT HISTORY: 
{context}

- FOLLOW UP QUESTION:
{follow_up_question}

- STANDALONE QUESTION:
"""},
]





def WEB_QUERY_REWRITE_TEMPLATE(query): 
    return [{"role": "user", "content": 
f"""You are a helpful assistant that rewrites a user query into a better query for web search.
The rewrite consists of two steps:
Step 1: Translate the query to English.
Step 2: Rewrites the translated query into a better query for web search.

Your response will be a JSON object with the following format, And you only need to give the final JSON object after rewrite.
Respond with the answer only. No justification or explanation.
{{"query": "the rewritten query"}}

---
{{Demonstration}} Query to rewrite: {query}
Query after rewrite:"""},
]

