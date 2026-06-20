# Pneuma's Command-Line Interface (CLI) Reference

## Table of Contents

1. [Getting Started](#getting-started)
    1. [Gain Access to Gated Models](#gain-access-to-gated-models)
    2. [Clone the Repository](#clone-the-repository)
    3. [Create a Conda Environment](#create-a-conda-environment)
    4. [Install Dependencies](#install-dependencies)
    5. [Download Required Models](#download-required-models)
    6. [(Optional) Download Datasets](#optional-download-datasets)
2. [Registration Module](#registration-module)
    1. [Setup](#setup)
    2. [Add Tables](#add-table)
    3. [Add Metadata](#add-metadata)
3. [Summarizer Module](#summarizer-module)
    1. [Summarize](#summarize)
4. [Index Generation Module](#index-generation-module)
5. [Query Module](#query-module)

## Getting Started

Below is some relevant information related to running Pneuma's CLI.

### Gain Access to Gated Models

If you want to access private/gated models such as `meta-llama/Meta-Llama-3-8B-Instruct`, you should create a user access token in HuggingFace. After that, you may login using the following commands:

```shell
pip install -U "huggingface_hub[cli]"
huggingface-cli login [your_token]
```

Alternatively, you may pass the token directly, as instructed in [Summarize](#summarize).

### Clone the repository

```shell
git clone https://github.com/TheDataStation/Pneuma
cd Pneuma
```

### Create a conda environment
```shell
conda create --name pneuma python=3.12 -y
conda activate pneuma
conda install -c nvidia cuda-toolkit -y
export CUDA_HOME=$CONDA_PREFIX
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
```

### Install dependencies

```shell
pip install -r requirements.txt
```

### Download datasets and benchmarks
```shell
python3 data_src/tables/downloader.py
python3 data_src/benchmarks/downloader.py
```

### Quick start on the ChEMBL Dataset
The following commands are used to setup Pneuma on ChEMBL dataset without context.
More detailed instructions on each modules can be found at [Registration Module](#registration-module) and below.

Go to the pneuma directory
```shell
cd pneuma
```

Register the tables
```shell
python3 registration/registration.py setup --db_path=out_chembl/chembl.db
python3 registration/registration.py add_tables --db_path=out_chembl/chembl.db ../data_src/tables/pneuma_chembl_10K SampleUser
```

Summarize the tables and generate indices
```shell
python3 summarizer/summarizer.py summarize --db_path=out_chembl/chembl.db
python3 index_generator/index_generator.py generate_index --db_path=out_chembl/chembl.db sample_index
```

Query the index
```shell
python3 query/query.py query --db_path=out_chembl/chembl.db sample_index "Design a data-driven framework to monitor and scrutinize the performance of account executives, enabling informed strategic choices and optimized resource allocation."
```

**Note: All commands in 'Example Usage' should be run from the `pneuma` directory.**

## Registration Module

This module is used to load data from various sources and context into DuckDB.

### Setup

**Usage**:

```shell
python3 registration.py setup --db_path=PATH/TO/DATABASE_NAME.db
```

**Description**: Initializes the database schema. Creates a DATABASE_NAME.db file in the specified path.

**Example Usage**:

```shell
python3 registration/registration.py setup --db_path=out/storage.db
```

### Add Table

**Usage**:

```shell
python3 registration/registration.py add_tables --db_path=PATH/TO/DATABASE_NAME.db [OPTION]... (PATH_TO_FOLDER/PATH_TO_FILE.(csv/parquet)) CREATOR_NAME
```

**Description**: Reads a table, formatted in CSV or PARQUET, from the local filesystem or an online storage bucket. If a file in a storage bucket is public, it can be read like a local file. The path of the table will be used as the ID.

- PATH_TO_FOLDER / PATH_TO_FILE can be a path in the local filesystem or a bucket URI. If a folder path is inserted, all files in the folder will be processed.
- CREATOR_NAME is the name of the person who runs this command.

**Options**:

- --s3_region=REGION

    Region of the s3 bucket.

- --s3_access_key=AWS_ACCESS_KEY

    AWS access key ID

- --s3_secret_access_key=AWS_SECRET_ACCESS_KEY

    AWS secret access key ID

**Examples Usage**:

```shell
python3 registration/registration.py add_tables --db_path=out/storage.db ../data_src/sample_data/csv/5cq6-qygt.csv SampleUser
```

### Add Metadata

**Usage**:

```shell
python3 registration/registration.py add_metadata --db_path=PATH/TO/DATABASE_NAME.db (PATH_TO_FOLDER/PATH_TO_FILE.txt) [context/summary] [TABLE_ID]
```

**Description**: Creates a context or summary entry for the specified table. If the metadata file is a CSV, TABLE_ID is not needed.

**Example Usage**:

```shell
python3 registration/registration.py add_metadata --db_path=out/storage.db ../data_src/sample_data/context/sample_context.txt context ../data_src/sample_data/csv/5cq6-qygt.csv
```

## Summarizer Module

The summarized module generates content summaries of registered tables, which will then be stored in an index. These summaries will be useful for answering users' content-related questions

### Summarize

**Usage**:

```shell
python3 summarizer.py summarize --db_path=PATH/TO/DATABASE_NAME.db [OPTION]... [TABLE_ID]
```

**Description**: Generates summary entries for the specified table. If TABLE_ID is not provided, generate summaries for all unsummarized tables.

**Options**:

- --hf_token: User access token from HuggingFace to access gated models. This option is not needed if you have been authenticated using huggingface-cli.

**Example Usage**:

```shell
python3 summarizer/summarizer.py summarize --db_path=out/storage.db ../data_src/sample_data/csv/5cq6-qygt.csv
```

## Index Generation Module

We store registered context and generated summaries as documents in a searchable (vector) index and keyword index, enabling the retrieval of the most relevant documents quickly and accurately. Given a set of tables, this module generates an index.

### Generate Index

**Usage**:

```shell
python3 index_generator.py generate_index --db_path=PATH/TO/DATABASE_NAME.db --index_path=PATH/TO/INDEXES/FOLDER INDEX_NAME ['TABLE_ID1','TABLEID2','TABLEID3',...]
```

**Description**: Generates an index with the name INDEX_NAME containing context and summary entries from the tables listed. If the list of tables is not provided, generate an index from all tables.

**Example Usage**:

```shell
python3 index_generator/index_generator.py generate_index --db_path=out/storage.db --index_path=out/indexes sample_index ../data_src/sample_data/csv/5cq6-qygt.csv
```

## Query Module

The query module answers users’ queries by searching through the index generated by the index generation module. This module receives a plain text query, retrieves the most relevant documents, and provides an answer. The retrieval process is done using hybrid search and a reranking algorithm is then performed.

### Query

**Usage**:

```shell
python3 query.py query --db_path=PATH/TO/DATABASE_NAME.db --index_path=PATH/TO/INDEXES/FOLDER INDEX_NAME QUERY [OPTIONS]
```

**Description**: Queries an index with name INDEX_NAME with the query QUERY. Returns a list of potentially relevant tables.

**Options**:

- --k=K

    The number of tables returned (default value: 1)

- --n=N

    A multiplicative factor that determines the number of documents to initially retrieve for each retriever.

- --alpha=ALPHA

    Weighing factor to prioritize keyword vs vector index. Alpha=0 means vector index results are ignored.

**Example Usage**:

```shell
python3 query/query.py query --db_path=out/storage.db --index_path=out/indexes sample_index "Why was this dataset created?"
```

