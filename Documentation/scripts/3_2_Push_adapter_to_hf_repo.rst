Pushing the adapter into HuggingFace
======================================


In this section, we will show you how to push your adapter to Hugging Face. This will allow you to easily access your adapter from the Hugging Face Hub and use it in your own projects.

.. note:: 
    This section assumes that you have already went through the fine-tuning using the Ludwig library and that you have a Hugging Face account. If you have not done so, please go back to this section first (2.3 fine-tuning using Ludwig).

Ludwig API
--------

Assuming that you have used Ludwig to fine-tune an open source llm of your choice. At the end of training, a new folder will be created called **results**. This is an example of what you will see after the training is done:

.. code-block:: bash

    results
    ├── api_experiment_run_x  # x refers to the training run (integer)
    │   ├── config.yaml
    │   ├── model
    │   │   ├── pytorch_model.bin
    │   │   ├── special_tokens_map.json
    │   │   ├── tokenizer_config.json
    │   │   ├── training_args.bin
    │   │   └── vocab.json
    │   ├── training.log
    │   └── training_results.json
    └── llm_adapter
        ├── config.json
        ├── pytorch_adapter.bin
        └── training_args.bin

Now to push your adapter to Hugging Face, you will need to run the following command:

.. code-block:: bash

    ludwig upload hf_hub \
        --repo_id {profile/repo_name} \
        --model_path /content/results/api_experiment_run

To be able to run this command, you will to create a Hugging Face token. To do so, go to your Hugging Face profile page and click on the **Access Tokens** button, then click on the **New token** button to generate your token. Then, copy the token and paste it in the console after running the command.

repo_id refers to the name of your repository on Hugging Face. For example, if you want to push your adapter to the following `repository <https://huggingface.co/Azulian/DoctorLLM2>`_, then the repo_id will be ``Azulian/DoctorLLM2``.

