# IFT6289-multimodal-rag
- Repository: https://github.com/JolineHe/IFT6289-multimodal-rag
- Dataset: https://huggingface.co/datasets/MongoDB/airbnb_embeddings
- MongoDB database: https://cloud.mongodb.com/v2/67dce45371a72379b3f838e7#/clusters/atlasSearch/6289NLP

## Setup variables in your virtual environment
create a .env with your corresponding HF_TOKEN, MONGODB_URI and OPENAI_API_KEY

## Setup database
Run ONLY ONCE:
```bash
python ./src/utils/data_ingestion.py
```
## Create indexes
Run ONLY ONCE:
```bash
python ./src/utils/indexing.py
```
We have created three indexes:
- 1 full text search index on xxx
- 1 vector search index on yyy
- 1 vector search index on zzz

## Run the app
```bash
python ./src/app.py
```