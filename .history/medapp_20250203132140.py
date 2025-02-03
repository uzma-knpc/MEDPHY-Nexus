from dotenv import load_dotenv
from langchain_core.tools import tool
import math
from datetime import datetime, timedelta
from langchain_core.tools import tool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import tool
from datetime import datetime
from datetime import datetime
from typing import Dict, Union
from datetime import datetime

#load 
load_dotenv()


llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=GOOGLE_API_KEY)

# Dictionary of dose limits (units in mSv)
OCCUPATIONAL_LIMITS = {
        "Annual dose limit to Workers": 50,  # mSv per year
        "Annual Exposure Limit to Workers (5-year avg)": 20,  # mSv per year
        "Monthly dose Limit to Worker": 4.16,  # mSv per month (approx.)
        "Lens of Eye Dose Limit to Worker": 50,  # mSv per year
        "Extremity Dose Limit to Worker": 500,  # mSv per year
        "Annual Dose Limit to Public": 1,  # mSv per year
        "Lens of Eye Dose Limit to Public": 15,  # mSv per year
        "Extremity Dose Limit to Public": 50,  # mSv per year
        "Annual Dose Limit to Caregivers": 5,  # mSv per year
        "Annual Dose Limit to Comforter": 5,  # mSv per year

    }
# # ✅ Tool 1: Unit Conversion
@tool
def unit_conversion_tool(value, from_unit, to_unit):
    """
    Converts radiation measurement units.
    """
    conversion_factors = {
        "Gy_to_rad": 100, "rad_to_Gy": 0.01,
        "Sv_to_rem": 100, "rem_to_Sv": 0.01,
        "Bq_to_Ci": 2.7e-11, "Ci_to_Bq": 3.7e10,
        "C/kg_to_R": 2.58e-4, "R_to_C/kg": 0.387e4,
    }
    key = f"{from_unit}_to_{to_unit}"
    if key in conversion_factors:
        result = value * conversion_factors[key]
        return f"**Conversion:** {value} {from_unit} → {to_unit}\n**Result:** {result:.4f} {to_unit}"
    else:
        return "⚠️ Invalid conversion request."
# ✅ Tool 2: Radioactive Decay Calculation

# ✅ Dictionary of Radioactive Nuclides & Their Half-Lives (in Days)
from datetime import datetime
import math
RADIOACTIVE_NUCLIDES = {
    "Uranium-238": 4.47e9 * 365,  # Billion years converted to days
    "Uranium-235": 704e6 * 365,   # Million years converted to days
    "Thorium-232": 14.05e9 * 365,
    "Plutonium-239": 24100,
    "Plutonium-238": 87.7,
    "Radon-222": 3.8,
    "Radium-226": 1600,
    "Polonium-210": 138,
    "Americium-241": 432,
    "Carbon-14": 5730,
    "Strontium-90": 28.8 * 365,
    "Iodine-131 (I-131)": 8.02,
    "I-131": 8.02,
    "Iodine-125(I-125)": 59.4,
    "I-125": 59.4,
    "Cesium-137": 30.2 * 365,
     "Cs-137": 30.2 * 365,
    "Technetium-99m": 0.25,  # 6 hours converted to days
    "Tc-99m": 0.25,  # 6 hours converted to days
    "Cobalt-60": 5.27 * 365,
    "Co-60": 5.27 * 365,
    "Cobalt-57": 271.8,
    "Co-57": 271.8,
    "Tritium (H-3)": 12.32 * 365,
    "Krypton-85": 10.76 * 365,
    "Ruthenium-106": 373.6,
    "Mollebdenium (Mo-99)": 66/24,

}
@tool
# ✅ Function to Calculate Radioactive Decay
def radioactive_decay_tool(nuclide, initial_activity, elapsed_days):
    """
    Computes remaining radioactive activity based on the nuclide, initial activity, and elapsed time.
    Generates a Medical Physics (MP) report in Google Doc style.

    Args:
        nuclide (str): Name of the radioactive nuclide.
        initial_activity (float): Initial activity in Bq or Ci or mCi.
        elapsed_days (float): Time elapsed in days .

    Returns:
        str: A formatted Google Doc-style report with computed decay results.
    """
    # Ensure valid inputs
    if initial_activity <= 0 or elapsed_days < 0:
        return "⚠️ Invalid input! Activity must be positive, and elapsed time cannot be negative."

    # Retrieve half-life from the dictionary
    half_life = RADIOACTIVE_NUCLIDES.get(nuclide, None)

    if half_life is None:
        return f"⚠️ Error: Nuclide '{nuclide}' not found in database. Please check spelling or add it."

    # Compute decay constant
    decay_constant = math.log(2) / half_life

    # Compute remaining activity
    remaining_activity = initial_activity * math.exp(-decay_constant * elapsed_days)

    # Generate Google Doc-style Report
    current_date = datetime.now().strftime("%Y-%m-%d")

    report = f"""
# 📊 **Medical Physics Report**
### 📅 Date: {current_date}
---

## **🔹 Radioactive Decay Calculation**
🔬 **Radionuclide:** {nuclide}  
📈 **Initial Activity:** {initial_activity} Bq  or Ci or mCi
⏳ **Half-life:** {half_life:.2f} days  
🕰️ **Elapsed Time:** {elapsed_days} days  
⚛️ **Remaining Activity:** {remaining_activity:.4f} Bq or Ci or mCi

---
🔚 **End of Report**
"""
#return report


 # ✅ Tool 3: print list of exposure limits
@tool
def print_exposure_limits(query: str = "all") -> str:
    """
    Returns a SNIF Google Doc–style report of exposure/dose limits for workers, public, or caregivers.

    Args:
        query (str): A string indicating which limits to display. Acceptable values:
                     "workers", "public", "caregivers","comforter" or "all" (default prints all limits).

    Returns:
        str: A formatted report with the requested dose limits.
    """
   
    query_lower = query.lower().strip()
    
    # Filter keys based on the query
    if query_lower == "workers":
        # Include keys related to workers (e.g., any key with "worker")
        keys = [k for k in OCCUPATIONAL_LIMITS if "worker" in k.lower()]
    elif query_lower == "public":
        keys = [k for k in OCCUPATIONAL_LIMITS if "public" in k.lower()]
    elif query_lower == "caregivers":
        keys = [k for k in OCCUPATIONAL_LIMITS if "caregiver" in k.lower()]
    elif query_lower == "comforter":
        keys = [k for k in OCCUPATIONAL_LIMITS if "comforter" in k.lower()]
   
    else:
        # If query is "all" or unrecognized, print all limits.
        keys = list(OCCUPATIONAL_LIMITS.keys())
    
    # Generate a SNIF Google Doc–style report
    current_date = datetime.now().strftime("%Y-%m-%d")
    report = f"""
# 📊 **Exposure/Dose Limits Report**
### 📅 Date: {current_date}
---

"""
    for key in keys:
        report += f"- **{key}**: {OCCUPATIONAL_LIMITS[key]} mSv\n"
    
    report += "\n---\n🔚 **End of Report**"
    return report


