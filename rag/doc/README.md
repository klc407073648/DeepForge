# 使用cursor生成RAG项目并跑通流程

## 提出问题

![image-20260419172833359](./images/提出问题.png)

## cursor给出具体方案

总体实现：

**目标**：可运行的最小可用 RAG：上传/索引知识库 → 基于检索回答问题（带引用片段）。

详细见  `独立_rag_从0到1_.md`


## 根据运行手册执行程序

详细参考 `运行手册.md`

中间遇到很多问题，具体见 `调试过程.md`

## 最终可执行程序

### 1. 安装并验证embeding模型可用

具体见 `安装并验证embeding模型可用.md`

### 2. 运行rag应用程序

![image-20260419181201173](./images/运行RAG程序.png)

### 3. 进行问答

![rag问答过程](./images/rag问答过程.gif)

## 整个代码流程图

```mermaid
flowchart TB
  subgraph startup [应用启动 lifespan]
    L1[FastAPI lifespan 开始]
    L2[get_chroma_client CHROMA_PERSIST_DIR]
    L3[get_or_create_collection COLLECTION_NAME cosine]
    L4[全局 _collection 就绪]
    L1 --> L2 --> L3 --> L4
  end

  subgraph deps [依赖注入 各路由]
    D1[get_settings 读 Settings]
    D2[get_collection 取 Chroma collection]
    D3[get_llm_client new OpenAICompatibleClient]
  end

  subgraph health [GET health]
    H1[_service_ready_detail]
    H2[chat: Key 是否配置]
    H3[embedding: local 则模型 id 非空 / http 则 embedding Key]
    H1 --> H2 --> H3
  end

  subgraph ingest [POST ingest 或 ingest batch]
    I1[require_embedding_config]
    I2[保存上传文件到 data/raw]
    I3[ingest_documents 到 ingest_paths]
    I4[load_document 读文本]
    I5[chunk_text 切块]
    I6[collection.delete where source]
    I7["_embed_in_batches batch_size = embedding_ingest_batch_size"]
    I8[embed_texts]
    I9[collection.add ids embeddings documents metadatas]
    I10[清理临时文件 返回 IngestResponse]
    I1 --> I2 --> I3 --> I4 --> I5 --> I6 --> I7 --> I8 --> I9 --> I10
  end

  subgraph embed [OpenAICompatibleClient.embed_texts]
    E0{texts 为空?}
    E1[返回空列表]
    E2{embedding_backend}
    subgraph localPath [local]
      Lm1[_ensure_local_model 锁 + to_thread SentenceTransformer]
      Lm2[_embed_local to_thread model.encode]
    end
    subgraph httpPath [http]
      Ht1[httpx POST base/embeddings]
      Ht2[解析 data 按 index 排序取 embedding]
    end
    E0 -->|是| E1
    E0 -->|否| E2
    E2 -->|local| Lm1 --> Lm2
    E2 -->|http| Ht1 --> Ht2
  end

  subgraph query [POST query]
    Q1[require_query_config 嵌入 + chat Key]
    Q2[search_relevant_chunks]
    Q3[embed_texts 单条 question]
    Q4[collection.query top_k]
    Q5{有 chunks?}
    Q6[返回无相关内容 QueryResponse]
    Q7[取 best distance 与 max_retrieval_distance 比较]
    Q8[过远则返回未找到可靠片段]
    Q9[build_messages]
    Q10[chat_completion HTTP]
    Q11[组装 Citations 返回 QueryResponse]
    Q1 --> Q2 --> Q3 --> Q4 --> Q5
    Q5 -->|否| Q6
    Q5 -->|是| Q7 --> Q8
    Q7 -->|通过| Q9 --> Q10 --> Q11
  end

  L4 --> D1
  D1 --> health
  D1 --> ingest
  D1 --> query
  D2 --> ingest
  D2 --> query
  D3 --> ingest
  D3 --> query
  I8 --> embed
  Q3 --> embed
```
