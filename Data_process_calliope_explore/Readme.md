
## REDAME

This code has been produced during the Energy-Climate Hackathon 21-25 June 2021. The filepaths and analysis will need to be updated. Part of the work has been done to run the Calliopen model (see https://calliope.readthedocs.io/en/stable/user/introduction.html) and to visualize data from the 'Adverse Weather Scenarios for Future Electricity Systems' created by Met Office.



## Contents

* Scripts for pre-process data to run Calliope (e.g. change time resolution):
   - Prepare_CF_AdverseWeather.ipynb
   - Transform_hour_to_day_CF_input_cal.ipynb
   - prepareCF.ipynb
   
* Scripts to run Calliope over UK, Ireland:
   - calliope_looping_input_data_reliab_postprocess.ipynb
   - change-storage-caps.ipynb
   - example-notebook.ipynb: general script to use calliope with pre-defined datasets
   - example-notebook_ADW.ipynb: script that uses the data from Adwerse Weather Scenarios created by Met Office.
   
* Scripts to explore the data:
   - Utils_explore.py: help functions to explore_data.ipynb
   - explore_data.ipynb: includes several ways to show extremes events and co-occurrence of events.
   
   
## Attribution

This work is derived from resources provided by organisers and participants of the hackathon. Credits to Laura Dawkins and Hannah Bloomfield for providing template code, and assistance with the data and models. Contributions to this work from Spyros Skarvelis-Kazakos, Bryn Pickering and Noelia Otero.
   
   
