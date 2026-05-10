PS D:\code\klc\test\DeepForge\rag> pip install sentence-transformers
Collecting sentence-transformers
  Downloading sentence_transformers-5.4.1-py3-none-any.whl.metadata (17 kB)
Requirement already satisfied: transformers<6.0.0,>=4.41.0 in D:\software\python\Lib\site-packages (from sentence-transformers) (4.42.4)
Requirement already satisfied: huggingface-hub>=0.23.0 in D:\software\python\Lib\site-packages (from sentence-transformers) (0.34.4)
Requirement already satisfied: torch>=1.11.0 in D:\software\python\Lib\site-packages (from sentence-transformers) (2.5.1)
Requirement already satisfied: numpy>=1.20.0 in D:\software\python\Lib\site-packages (from sentence-transformers) (1.26.4)
Requirement already satisfied: scikit-learn>=0.22.0 in D:\software\python\Lib\site-packages (from sentence-transformers) (1.7.1)
Requirement already satisfied: scipy>=1.0.0 in D:\software\python\Lib\site-packages (from sentence-transformers) (1.16.1)
Requirement already satisfied: typing_extensions>=4.5.0 in D:\software\python\Lib\site-packages (from sentence-transformers) (4.15.0)
Requirement already satisfied: tqdm>=4.0.0 in D:\software\python\Lib\site-packages (from sentence-transformers) (4.67.0)
Requirement already satisfied: filelock in D:\software\python\Lib\site-packages (from transformers<6.0.0,>=4.41.0->sentence-transformers) (3.16.1)
Requirement already satisfied: packaging>=20.0 in D:\software\python\Lib\site-packages (from transformers<6.0.0,>=4.41.0->sentence-transformers) (26.0)
Requirement already satisfied: pyyaml>=5.1 in D:\software\python\Lib\site-packages (from transformers<6.0.0,>=4.41.0->sentence-transformers) (6.0.2)
Requirement already satisfied: regex!=2019.12.17 in D:\software\python\Lib\site-packages (from transformers<6.0.0,>=4.41.0->sentence-transformers) (2024.11.6)
Requirement already satisfied: requests in D:\software\python\Lib\site-packages (from transformers<6.0.0,>=4.41.0->sentence-transformers) (2.32.3)
Requirement already satisfied: safetensors>=0.4.1 in D:\software\python\Lib\site-packages (from transformers<6.0.0,>=4.41.0->sentence-transformers) (0.4.5)
Requirement already satisfied: tokenizers<0.20,>=0.19 in D:\software\python\Lib\site-packages (from transformers<6.0.0,>=4.41.0->sentence-transformers) (0.19.1)
Requirement already satisfied: fsspec>=2023.5.0 in D:\software\python\Lib\site-packages (from huggingface-hub>=0.23.0->sentence-transformers) (2024.10.0)
Requirement already satisfied: joblib>=1.2.0 in D:\software\python\Lib\site-packages (from scikit-learn>=0.22.0->sentence-transformers) (1.5.1)
Requirement already satisfied: threadpoolctl>=3.1.0 in D:\software\python\Lib\site-packages (from scikit-learn>=0.22.0->sentence-transformers) (3.6.0)
Requirement already satisfied: networkx in D:\software\python\Lib\site-packages (from torch>=1.11.0->sentence-transformers) (3.4.2)
Requirement already satisfied: jinja2 in D:\software\python\Lib\site-packages (from torch>=1.11.0->sentence-transformers) (3.1.6)
Requirement already satisfied: setuptools in D:\software\python\Lib\site-packages (from torch>=1.11.0->sentence-transformers) (74.0.0)
Requirement already satisfied: sympy==1.13.1 in D:\software\python\Lib\site-packages (from torch>=1.11.0->sentence-transformers) (1.13.1)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in D:\software\python\Lib\site-packages (from sympy==1.13.1->torch>=1.11.0->sentence-transformers) (1.3.0)
Requirement already satisfied: colorama in D:\software\python\Lib\site-packages (from tqdm>=4.0.0->sentence-transformers) (0.4.6)
Requirement already satisfied: MarkupSafe>=2.0 in D:\software\python\Lib\site-packages (from jinja2->torch>=1.11.0->sentence-transformers) (3.0.3)
Requirement already satisfied: charset-normalizer<4,>=2 in D:\software\python\Lib\site-packages (from requests->transformers<6.0.0,>=4.41.0->sentence-transformers) (3.4.0)
Requirement already satisfied: idna<4,>=2.5 in D:\software\python\Lib\site-packages (from requests->transformers<6.0.0,>=4.41.0->sentence-transformers) (2.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in D:\software\python\Lib\site-packages (from requests->transformers<6.0.0,>=4.41.0->sentence-transformers) (2.2.3)
Requirement already satisfied: certifi>=2017.4.17 in D:\software\python\Lib\site-packages (from requests->transformers<6.0.0,>=4.41.0->sentence-transformers) (2024.8.30)
Downloading sentence_transformers-5.4.1-py3-none-any.whl (571 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 571.3/571.3 kB 5.0 MB/s  0:00:00
Installing collected packages: sentence-transformers
Successfully installed sentence-transformers-5.4.1
PS D:\code\klc\test\DeepForge\rag> python scripts\verify_local_embedding.py
modules.json: 100%|████████████████████████████████████████████████████████████████████████████████████████| 229/229 [00:00<?, ?B/s]
D:\software\python\Lib\site-packages\huggingface_hub\file_download.py:143: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\summer\.cache\huggingface\hub\models--sentence-transformers--paraphrase-MiniLM-L6-v2. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
config_sentence_transformers.json: 100%|███████████████████████████████████████████████████████████████████| 122/122 [00:00<?, ?B/s]
README.md: 3.51kB [00:00, ?B/s]
sentence_bert_config.json: 100%|█████████████████████████████████████████████████████████████████████████| 53.0/53.0 [00:00<?, ?B/s]
config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 629/629 [00:00<?, ?B/s]
Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
model.safetensors: 100%|███████████████████████████████████████████████████████████████████████| 90.9M/90.9M [00:03<00:00, 28.8MB/s]
tokenizer_config.json: 100%|███████████████████████████████████████████████████████████████████████████████| 314/314 [00:00<?, ?B/s]
vocab.txt: 232kB [00:00, 2.53MB/s]
tokenizer.json: 466kB [00:00, 7.43MB/s]
special_tokens_map.json: 100%|█████████████████████████████████████████████████████████████████████████████| 112/112 [00:00<?, ?B/s]
config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 190/190 [00:00<?, ?B/s]
dims: 2 x 384
ok