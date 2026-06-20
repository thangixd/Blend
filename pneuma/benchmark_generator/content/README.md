## 1. Setup
Python 3.8 or higher.
```
pip install -r requirements.txt 
```

## 2. Import data
Create a folder named `data` side by side with the repo folder `content` (by default)

Create a folder for each dataset in `data`, e.g. `nyc_open`

Create a folder named `input_files` in each dataset folder. 

```
    data
        nyc_open
            input_files
```
Put csv files in `input_files`. 
To assign a ID for each tabble(csv), use the file name format:
```
<table_caption>_SEP_<table_ID>.csv
``` 
Othewise the table ID will be assigned automatically.
Run
```
./import_tables.sh <dataset>
```
To import data

## 3. Generate questions
3.1) Set OpenAI key
```
export OPENAI_API_KEY=<key>
```
3.2) Run
```
./gen_questions.sh <dataset> <number of questions>
```
Questions are output to `./output/<dataset>/questions.jsonl`

Prompt logs are in file `prompt/log/<dataset>/log_...`. You can copy the Table Caption and Table Data parts in the file to ChatGPT and then ask ChatGPT to format it.

```
Table Caption:
...
Table Data:
...
```
## 4. Annotate answer tables
Run
```
./annotate.sh <dataset>
```
The output quations is at `./output/<dataset>/<dataset>_questions_annotated.jsonl`.
Each line of this file is a json object:
```
"question": the question text
"answer_tables": identifiers of answer tables
```