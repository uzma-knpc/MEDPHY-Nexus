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

