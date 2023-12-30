Preparing and Loading the 'doctorllm' dataset for LLM Adapter Training
=========================================================

In this documentation, we'll outline the process of preparing the 'doctorllm' dataset used for training an LLM adapter. This dataset is a combination of two distinct datasets from Hugging Face, which were concatenated and processed using the pandas library.

.. note:: 
   *While we've found and used more comprehensive and superior datasets for adapter training, this tutorial utilizes these particular two for their simplicity and straightforward preprocessing requirements. This choice allows us to focus on the methodology rather than data processing complexities.*



Prerequisites
-------------

Before you start, ensure you have the following:

- An account on HuggingFace. You can create one `here <https://huggingface.co/>`_.
- The datasets library installed: `pip install datasets`.
- The pandas library installed: `pip install pandas`.

Datasets Used
-------------

We used two datasets for this process:

1. LinhDuong/chatdoctor-5k: `https://huggingface.co/datasets/LinhDuong/chatdoctor-5k/viewer/default`
2. mrm8488/chatdoctor200k: `https://huggingface.co/datasets/mrm8488/chatdoctor200k/viewer/default/train`

.. figure:: /Documentation/images/docdata0.PNG
   :width: 80%
   :align: center
   :alt: Screenshot of the first dataset
   :name: dataset1_preview

   Preview of the LinhDuong/chatdoctor-5k dataset.

.. figure:: /Documentation/images/docdata1.PNG
   :width: 80%
   :align: center
   :alt: Screenshot of the second dataset
   :name: dataset2_preview

   Preview of the mrm8488/chatdoctor200k dataset.


Data Concatenation and Processing
---------------------------------

The datasets were concatenated, and the only processing required was swapping column values within the dataframe (Instruct and Input columns). Below is the code snippet illustrating this process:

.. code-block:: python

    from datasets import load_dataset, concatenate_datasets
    import pandas as pd

    # Load datasets
    dataset1 = load_dataset("LinhDuong/chatdoctor-5k", split='train')
    dataset2 = load_dataset("mrm8488/chatdoctor200k", split='train')

    # Select a consistent number of samples from each dataset
    dataset2 = dataset2.select(range(5000))

    # Concatenate datasets
    dataset = concatenate_datasets([dataset1, dataset2])

    # Convert to pandas DataFrame
    df = dataset.to_pandas()

    # Swap column values
    df['instruction'], df['input'] = df['input'].copy(), df['instruction'].copy()

    # Save your processed DataFrame if needed
    df.to_csv('processed_doctorllm.csv', index=False)


