def ROUTER_TEMPLATE(query):
    return [{"role": "user", "content":
f"""You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to "Ladder" and "NOM", a tool designed to build visualize artificial neural networks.
Use the vectorstore for questions on these topics. Otherwise, use web search.

Your response will be a JSON object with the following format.
{{"tool": "vectorstore" or "web search"}}

---
### Examples:

Question: 該怎麼在建構模型時修改表格?
Answer: {{"tool": "vectorstore"}}

Question: advantage of LSTM over RNN
Answer: {{"tool": "web search"}}

Question: Can I extend Ladder to work with other machine learning frameworks besides TensorFlow?
Answer: {{"tool": "vectorstore"}}

Question: How do I visualize the structure of my model?
Answer: {{"tool": "vectorstore"}}

Question: What is the current state-of-the-art model in NLP?
Answer: {{"tool": "web search"}}

Question: Does Ladder support ReLU and sigmoid activation functions?
Answer: {{"tool": "vectorstore"}}

Question: What is a convolution layer in machine learning?
Answer: {{"tool": "web search"}}

Question: 如何加入Dropout層?
Answer: {{"tool": "vectorstore"}}

Question: What are the benefits of using ensemble learning?
Answer: {{"tool": "web search"}}

Question: Does Ladder provide pre-built templates for CNN models?
Answer: {{"tool": "vectorstore"}}

Question: What are the latest updates on OpenAI GPT models?
Answer: {{"tool": "web search"}}

Question: How do I share a model built with other users?
Answer: {{"tool": "vectorstore"}}

Question: 描述Ladder模型開發的工作流程
Answer: {{"tool": "vectorstore"}}

Question: What's the difference between logistic regression and linear regression?
Answer: {{"tool": "web search"}}

Question: How do I debug training issues in Ladder?
Answer: {{"tool": "vectorstore"}}

Question: What are the top machine learning conferences happening in 2024?
Answer: {{"tool": "web search"}}

Question: Can Ladder be used to fine-tune pretrained models?
Answer: {{"tool": "vectorstore"}}

Question: Is dropout useful in recurrent neural networks?
Answer: {{"tool": "web search"}}

Question: How can I add a custom activation function in Ladder?
Answer: {{"tool": "vectorstore"}}

Question: What are the pros and cons of using decision trees for classification?
Answer: {{"tool": "web search"}}

Question: 請說明監督式學習和無監督式學習之間的差異。
Answer: {{"tool": "web search"}}

Question: What activation functions are available?
Answer: {{"tool": "vectorstore"}}

Question: machine learning cross-validation
Answer: {{"tool": "web search"}}

Question: What is Ladder, and what is its main purpose?
Answer: {{"tool": "vectorstore"}}

Question: SELU 激活函數在什麼情況下特別有利？
Answer: {{"tool": "web search"}}

Question: 什麼是NOM, 它與 Ladder 有什麼關係？
Answer: {{"tool": "vectorstore"}}

Question: How can i add a new hidden layer?
Answer: {{"tool": "vectorstore"}}

Question: What is the purpose of the Collector layer?
Answer: {{"tool": "web search"}}

Question: 如何上傳數據表格？
Answer: {{"tool": "vectorstore"}}

Question: 使用交叉驗證時, 應該如何選擇K值?
Answer: {{"tool": "web search"}}

Question: result analysis after training
Answer: {{"tool": "vectorstore"}}

Question: LSTM 門控機制
Answer: {{"tool": "web search"}}

Question: 保存專案檔案格式是什麼
Answer: {{"tool": "vectorstore"}}

Question: 翻譯成中文
Answer: {{"tool": "web search"}}

Question: Can I export a trained model?
Answer: {{"tool": "vectorstore"}}

Question: What file formats does Ladder support for importing datasets
Answer: {{"tool": "vectorstore"}}

Question: What is the latest breakthrough in reinforcement learning?
Answer: {{"tool": "web search"}}

Question: Does Ladder support automatic model optimization?
Answer: {{"tool": "vectorstore"}}

Question: What is transfer learning, and when should I use it
Answer: {{"tool": "web search"}}

Question: How do I integrate Ladder with my existing machine learning pipeline?
Answer: {{"tool": "vectorstore"}}

Question: Can NOM be used to build recurrent neural networks?
Answer: {{"tool": "vectorstore"}}

Question: What is the difference between Ladder and other deep learning platforms
Answer: {{"tool": "vectorstore"}}

Question: How can I visualize weight distributions?
Answer: {{"tool": "vectorstore"}}

Question: How do I debug a model that is not converging?
Answer: {{"tool": "web search"}}

Question: How do I evaluate model accuracy?
Answer: {{"tool": "vectorstore"}}

Question: What are the pre-built templates available in Ladder
Answer: {{"tool": "vectorstore"}}

Question: Does Ladder provide any debugging tools for model training?
Answer: {{"tool": "vectorstore"}}

Question: Which machine learning algorithms work best for time series forecasting
Answer: {{"tool": "web search"}}

Question: Does this tool provide built-in regularization techniques?
Answer: {{"tool": "vectorstore"}}

Question: What are the key components of a Ladder model development workflow?
Answer: {{"tool": "vectorstore"}}

Question: What is the difference between Adam and SGD optimizers
Answer: {{"tool": "web search"}}

Question: What loss functions can I use?
Answer: {{"tool": "vectorstore"}}

Question: What are some real-world applications of GANs?
Answer: {{"tool": "web search"}}

Question: Does the platform support multi-GPU training
Answer: {{"tool": "vectorstore"}}

Question: How do I debug a model that is not converging?
Answer: {{"tool": "web search"}}

Question: Is there a way to automate hyperparameter tuning
Answer: {{"tool": "vectorstore"}}

Question: Can I train a reinforcement learning model?
Answer: {{"tool": "vectorstore"}}

Question: How does batch normalization improve deep learning models
Answer: {{"tool": "web search"}}

Question: How does the Transformer architecture compare to CNNs
Answer: {{"tool": "web search"}}

Question: How can I track model training progress?
Answer: {{"tool": "vectorstore"}}

Question: What are the top AI research conferences in 2025?
Answer: {{"tool": "web search"}}

Question: How do I add a new layer to my model?
Answer: {{"tool": "vectorstore"}}

Question: Explain the concept of vanishing gradients in deep learning.
Answer: {{"tool": "web search"}}

Question: What is the best way to optimize model performance
Answer: {{"tool": "web search"}}

Question: What are some best practices for hyperparameter tuning?
Answer: {{"tool": "web search"}}

Question: Can I export model weights?
Answer: {{"tool": "vectorstore"}}

Question: Can I use pre-trained models in my project
Answer: {{"tool": "vectorstore"}}

Question: What is the recommended batch size?
Answer: {{"tool": "web search"}}

Question: Can I visualize my model's architecture?
Answer: {{"tool": "vectorstore"}}

Question: How do I handle imbalanced data in training?
Answer: {{"tool": "web search"}}

Question: How do I share my trained model with others
Answer: {{"tool": "vectorstore"}}

Question: How do I integrate external datasets?
Answer: {{"tool": "vectorstore"}}

Question: What do activation functions do?
Answer: {{"tool": "web search"}}

Question: Can I run my model on edge devices?
Answer: {{"tool": "vectorstore"}}

Question: Tell me about an example usage of CNN model
Answer: {{"tool": "web search"}}
---
### {{Demonstration}} 
Question: {query}
Answer: """},
]





def DB_ROUTER_TEMPLATE(source_knowledge, query): 
    return [{"role": "user", "content":
f"""The following retrieved content may contain information about Ladder, a beginner-friendly tool for building artificial neural networks.
Your task is to determine whether the content is relevant and necessary for answering the user's query regarding Ladder.
Note:
1. The focus should be on whether the content provides useful or required information related to Ladder's platform, features, or usage.
2. This is a True or False question. Respond only with "Yes" if the content is relevant and necessary, or "No" if it is not.

---
### contents:
{source_knowledge}
### query:
{query}"""},
]





def WEB_ROUTER_TEMPLATE(query):
    return [{"role": "user", "content": 
f"""You are an helpful ai assistant. You have access to a search tool. Use the search tool in the following circumstances:
- When the question pertains to current events or requires real-time information (e.g., news, researchs, specific dates).
- When the question involves a term you are unfamiliar with.
- When the question explicitly asks you to browse or provide links to references.

Determine whether the search tool is needed to answer the given question. Respond with only 'yes' or 'no'.
---
### Examples:
Question: Search for methods to prevent overfitting
Answer: yes

Question: advantage of LSTM over RNN
Answer: no

Question: recent breakthroughs in machine learning
Answer: yes

Question: Latest advancements in reinforcement learning algorithms
Answer: yes

Question: Differences between gradient descent and stochastic gradient descent
Answer: no

Question: What is the difference between supervised and unsupervised learning?
Answer: no

Question: Explain the vanishing gradient problem in deep learning
Answer: no

Question: Research papers on explainable AI published in 2024
Answer: yes

Question: Applications of convolutional neural networks in image processing
Answer: no

Question: News about OpenAI's recent product launches
Answer: yes

Question: Describe how decision trees work in machine learning
Answer: no

Question: latest Kaggle competition on NLP
Answer: yes

Question: Why are ReLU activations preferred over sigmoid in deep learning?
Answer: no

Question: List top machine learning conferences happening this year
Answer: yes

Question: How does batch normalization help in training deep neural networks?
Answer: no

Question: When was the latest version of Python released?
Answer: yes

Question: What are the ethical implications of AI-generated content?
Answer: no

Question: 用繁體中文回答
Answer: no

Question: Find the best libraries for handling missing data in Python in 2024
Answer: yes

Question: Differences between K-means and DBSCAN clustering algorithms
Answer: no

Question: Search for studies on the impact of AI in healthcare in 2024
Answer: yes

Question: Latest trends in transfer learning techniques
Answer: yes

Question: How does dropout regularization work?
Answer: no

Question: What is the current state-of-the-art model for text-to-image generation in 2024?
Answer: yes

Question: Explain the concept of overfitting in machine learning
Answer: no

Question: Find the top-performing transformer-based architectures in 2024
Answer: yes

Question: What are the main components of a support vector machine (SVM)?
Answer: no

Question: Search for recent implementations of reinforcement learning in robotics
Answer: yes

Question: Describe how gradient boosting works
Answer: no

Question: New algorithms for feature selection introduced in 2024
Answer: yes

Question: Differences between AdaBoost and XGBoost
Answer: no

Question: Best practices for deploying ML models in 2024
Answer: yes

Question: Explain the purpose of cross-validation in model evaluation
Answer: no

Question: Search the internet for the latest updates on federated learning frameworks
Answer: yes

Question: What are the advantages of convolutional layers in CNNs?
Answer: no

Question: Who won the latest Kaggle competition on NLP?
Answer: yes

Question: Applications of PCA in dimensionality reduction
Answer: no

Question: Find examples of data augmentation techniques for small datasets
Answer: yes

Question: How does the attention mechanism work in transformers?
Answer: no

Question: 翻譯成中文
Answer: no

---
### {{Demonstrate}}:
Question: {query}
Answer: """},
]

