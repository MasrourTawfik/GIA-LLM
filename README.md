# GIA-LLM

## Description

## Key Features


## Getting Started
To get started with this project, follow these steps:

1. Mak sure you have Git installed in your computer, if it is not installed, then download it from [here](https://git-scm.com/downloads).
2. Open the terminal.
3. Clone the repository: `git clone https://github.com/MasrourTawfik/GIA-LLM.git`

## Usage
To learn how to use the interface, follow these steps:

1. Navigate to the chainlit directory.
2. Create a virtual env using the following command `conda env -n env_name python=3.10`
3. Download the model from [here](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main) and place it in the same directory.
4. Inside the `app.py` file, change the `local_llm` variable to store the name of the model you downloaded, as an example `local_llm = "./mistral-7b-instruct-v0.1.Q4_K_S.gguf"`.
5. Run the following command to install the dependencies: `pip install -r requirements.txt`
6. Execute the following command: `chainlit run app.py -w`
7. Open your browser and go to `localhost:8080`

