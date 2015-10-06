# How to update metadata

(to be moved to the documentation)

## 1. configure auto-sklearn on all datasets

    python 01_autosklearn_create_metadata_runs.py --datasets datasets_test.csv
    --runs-per-dataset 1 --output-directory /tmp/metadata --time-limit 360 --per-run-time-limit 60 --ml-memory-limit 3072

## 2. get the test performance of these configurations

    bash 02_validate_autosklearn_metadata_runs.sh /tmp/metadata

## 3. convert smac-validate output into aslib format

    python 03_autosklearn_retrieve_metadata.py /tmp/metadata /tmp/metadata/metadata/
    --num-runs 1 --only-best True

## 4. calculate metafeatures

    python 04_autosklearn_calculate_metafeatures.py datasets_test.csv /tmp/metadata/metafeatures --memory-limit 3072

## 5. create aslib files

    python 05_autosklearn_create_aslib_files.py /tmp/metadata/metafeatures
    /tmp/metadata/metadata /tmp/metadata/aslib name 60 3072
