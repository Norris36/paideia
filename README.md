# dcei-gpt-exp

Internal site to demo GPT capabilities, running on GN Azure infrastucture


# How to run
Rename .env.template to .env and fill in the missing values

    # Create virtual environment (only done first time)
    python -m venv venv

    # Activate virual environment, install dependicies and run script
    .\venv\Scripts\activate
    pip install -r .\requirements.txt
    streamlit run .\gpt-explorer.py
    




# How to update enviroment

If you have added new packages with pip, remember to update the environment:

    pip freeze > .\requirements.txt
