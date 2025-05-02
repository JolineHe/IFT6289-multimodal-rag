# IFT6289-multimodal-rag
- Repository: https://github.com/JolineHe/IFT6289-multimodal-rag
- Dataset: https://huggingface.co/datasets/MongoDB/airbnb_embeddings
- MongoDB database: https://cloud.mongodb.com/v2/67dce45371a72379b3f838e7#/clusters/atlasSearch/6289NLP

## Setup variables in your virtual environment
create .env from .env.template and put the values.

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

## Run the app
```bash
python ./app/my_app.py
```