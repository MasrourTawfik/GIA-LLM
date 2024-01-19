
Data Preparation and Upload Notebook Documentation
==================================================

Overview
--------
This document details the code and its functionalities in the Jupyter Notebook designed to prepare and upload a dataset to Hugging Face's hub.

.. contents::
   :local:

Unzipping Data
--------------

.. code-block:: python

    !unzip /content/data_LLM.zip

**Comment:**
Unzips the 'data_LLM.zip' file, ensuring the raw data is accessible for processing.

Installing Datasets Package
---------------------------

.. code-block:: python

    !pip install datasets

**Comment:**
Installs the 'datasets' package necessary for efficient data handling and processing in Python.

Merging JSON Files into JSONL
-----------------------------

.. code-block:: python

    import os
    import json
    import glob

    directory = "/content/data_LLM"
    output_jsonl_filename = "merged_dataset.jsonl"
    json_pattern = os.path.join(directory, '*.json')
    file_list = glob.glob(json_pattern)

    with open(output_jsonl_filename, 'w') as outfile:
        for file in file_list:
            with open(file, 'r') as f:
                json_obj = json.load(f)
                outfile.write(json.dumps(json_obj) + '\n')

**Comment:**
Reads multiple JSON files from the specified directory and merges them into a single JSONL file, creating a unified dataset structure.

Loading Dataset
---------------

.. code-block:: python

    from datasets import Dataset, Features, Value, ClassLabel, Sequence, load_dataset

    jsonl_file_path = output_jsonl_filename
    dataset = load_dataset('json', data_files=jsonl_file_path)
    print(dataset['train'][0])

**Comment:**
Loads the merged JSONL file as a dataset using the 'datasets' library and prints the first entry for verification.

Authentication for Hugging Face
-------------------------------

.. code-block:: python

    from huggingface_hub import notebook_login
    notebook_login()

**Comment:**
Prompts for Hugging Face authentication, ensuring secure access for uploading the dataset.

Pushing to Hugging Face
-----------------------

.. code-block:: python

    dataset.push_to_hub("badreddine_LLM_data")

**Comment:**
Pushes the prepared dataset to Hugging Face's hub under the specified repository name, making it available for global access.

Conclusion
----------
This document provided a step-by-step guide to the notebook's process for preparing and uploading a dataset to Hugging Face's hub.
