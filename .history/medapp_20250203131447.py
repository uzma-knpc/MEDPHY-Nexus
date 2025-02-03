from dotenv import load_dotenv
from langchain_core.tools import tool
import math
from datetime import datetime, timedelta
from langchain_core.tools import tool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load 
load_dotenv()


llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=GOOGLE_API_KEY)

# Tool-1: ✅ Unit Conversion Function
@tool  #wrapper (function is wrapped into class by calling class)
def convert_units(value, from_unit, to_unit):
    """
    Converts radiation measurement units.Generates a Medical Physics (MP) report with formatted results.

    Args:
        value (float): The numerical value to convert.
        from_unit (str): The original unit (e.g., 'Gy', 'rad').
        to_unit (str): The target unit (e.g., 'rad', 'Gy').

    Returns:
        float or str: Converted value if valid, else error message.
        ## **1️⃣ Unit Conversion**
      **📌 Conversion Requested:** 1 Gy → rad  
      **📍 Result:** rad  

---
    """
    conversion_factors = {
        "Gy_to_rad": 100, "rad_to_Gy": 0.01,
        "Sv_to_rem": 100, "rem_to_Sv": 0.01,
        "Bq_to_Ci": 2.7e-11, "Ci_to_Bq": 3.7e10
    }
    key = f"{from_unit}_to_{to_unit}"
    if key in conversion_factors:
      unit_conversion_result = convert_units(1, "Gy", "rad")
      return value * conversion_factors[key]
    else:
        return "Invalid conversion units."
    #TOOL-2: ✅ Radioactive Decay Calculation
@tool
def calculate_activity(initial_activity, half_life, elapsed_days):
  
    """
    Computes remaining radioactive activity after a given time.Generates a Medical Physics (MP) report with formatted results.


    Args:
        initial_activity (float): Initial activity in Bq or Ci.
        half_life (float): Half-life of the radionuclide in days.
        elapsed_days (float): Time elapsed in days.

    Returns:
        float or str: Remaining activity or error message.

        ## **2️⃣ Radioactive Decay Calculation**
        **📌 Input:**  
        - **Initial Activity:** 100 Bq  
        - **Half-life:** 8 days  
        - **Time Elapsed:** 20 days  

        **📍 Result:**  
        - **Remaining Activity:**  Bq  

---
    """
    if initial_activity <= 0 or half_life <= 0:
        return "Invalid input values. Activity and half-life must be positive."
    
    decay_constant = math.log(2) / half_life
    remaining_activity = initial_activity * math.exp(-decay_constant * elapsed_days)
    return round(remaining_activity, 4)  # Rounded for readability


