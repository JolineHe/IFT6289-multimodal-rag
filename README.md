# IFT6289-multimodal-rag
https://github.com/JolineHe/IFT6289-multimodal-rag
Database: https://huggingface.co/datasets/MongoDB/airbnb_embeddings
Stored database in MongoDB: https://cloud.mongodb.com/v2/67dce45371a72379b3f838e7#/clusters/atlasSearch/6289NLP
## Set dataset
1. Setup .env with your corresponding HF_TOKEN, MONGODB_URI and OPENAI_API_KEY
2. Run the following:
```bash
python ./src/data_ingest.py
```

## Run the app
```bash
python ./src/app.py
```