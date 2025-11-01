# Streamlit UTBK Portfolio

This project contains a Streamlit app to present analysis and model predictions for a UTBK scores dataset.

Files created:
- app.py : Streamlit application
- train_model.py : Training script
- models/ : contains reg_model.pkl and class_model.pkl (saved in this environment)
- NILAI UTBK ANGK 4.xlsx : original dataset (packaged by user)
- requirements.txt : dependencies

How to run locally:
1. Create a virtual environment and install dependencies from requirements.txt
2. Place `NILAI UTBK ANGK 4.xlsx` in the project root
3. (Optional) Run `python train_model.py` to re-train models and save into `models/`
4. Run `streamlit run app.py`

Note: On Streamlit Cloud, create a new repo with these files and set `app.py` as the main file.
