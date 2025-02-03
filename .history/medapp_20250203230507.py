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
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime, timedelta
import math
import os
#load 
load_dotenv()
secret_key = os.environ.get("GOOGLE_API_KEY")
#print(f"SECRET_KEY: {secret_key}")
llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=secret_key)


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
#  ✅ Tool 1: Unit Conversion
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
    "Mollebdenium (MO-99)": 66/24,
}
# ✅ Tool 2: Radioactive Decay Calculation
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
 # ✅ Tool 4:patient release criteria
@tool
def patient_release_decision(neck_dose_microSv_hr: float, exposure_duration_hr: float, sef: str) -> str:
    """
    Evaluates whether a patient treated with I-131 can be released based on their neck dose,
    exposure duration, and Socio-Economic Factor (SEF). Generates a structured SNIF Google Doc-style report.
    
    Args:
        neck_dose_microSv_hr (float): Measured neck dose in μSv/hr.
        exposure_duration_hr (float): Duration of exposure (hours) at the measured dose rate.
        sef (str): Socio-Economic Factor, which should be either "good" or "bad".
        
    Returns:
        str: A formatted SNIF Google Doc-style report with the calculated TEDE and release recommendation.
    """
    # Validate input (basic validation)
    if neck_dose_microSv_hr < 0: #or exposure_duration_hr < 0:
        return "⚠️ Invalid input! Neck dose and exposure duration must be positive numbers."
    
    sef = sef.lower().strip()
    if sef not in ["good", "bad"]:
        return "⚠️ Invalid SEF! Please provide 'GOOD' or 'BAD' for the socio-economic factor."
    
    # Calculate TEDE:
    # Total dose in μSv = neck_dose_microSv_hr * exposure_duration_hr
    # Convert μSv to mSv: divide by 1,000.
    # Convert mSv to rem: multiply by 0.1 (since 1 mSv = 0.1 rem)
    total_dose_microSv = neck_dose_microSv_hr * 1.44*24*8.02*0.25
    total_dose_mSv = total_dose_microSv / 1000
    tede_rem = total_dose_mSv * 0.1
    print(total_dose_microSv)
    print(tede_rem)
    # Define the discharge threshold (0.5 rem)
    discharge_threshold = 0.2
    # standard is 0.5
    # Determine recommendation based on TEDE and SEF.
    if tede_rem < discharge_threshold:
        if sef == "good":
            recommendation = "Immediate release recommended."
        else:
            recommendation = "Patient discharge can be considered, but a prolonged hospital stay is advised due to socio-economic factors."
    else:
        recommendation = "Patient should remain hospitalized until the dose decays to acceptable levels."
#- **Exposure Duration:** {exposure_duration_hr:.2f} hours  
    # Generate SNIF Google Doc-style report.
    current_date = datetime.now().strftime("%Y-%m-%d")
    report = f"""
# 📊 **Medical Physics Patient Release Report**
### 📅 Date: {current_date}
---

## **🔹 Patient Exposure Assessment**

- **Neck Dose Rate:** {neck_dose_microSv_hr:.2f} μSv/hr  
- **Calculated Total Dose:** {total_dose_microSv:.2f} mSv  
- **Total Effective Dose Equivalent (TEDE):** {tede_rem:.3f} rem  

## **🔸 Socio-Economic Factor (SEF)**
- **SEF:** {sef.capitalize()}

## **🔹 Discharge Recommendation**
- **Threshold for Release:** 0.5 rem  
- **Occupancy factor:**0.25
- **Half life(I-131):**8.02 days
- **Decision:** {recommendation}

---
🔚 **End of Report**
"""
    return report


# Tool-5:✅ AI Agent to Call Tools
@tool
def medical_physics_agent(query):
    """
    An AI agent that takes user queries and calls the appropriate MP tool.
    """
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Identify Query Type
    if "convert" in query:
        value = float(query.split()[1])  # Extract first number
        from_unit, to_unit = query.split()[2], query.split()[4]  # Extract units
        result = unit_conversion_tool(value, from_unit, to_unit)
        title = "Unit Conversion"
    
    elif "decay" in query:
        values = [float(i) for i in query.split() if i.replace('.', '', 1).isdigit()]
        if len(values) == 3:
            result = radioactive_decay_tool(values[0], values[1], values[2])
            title = "Radioactive Decay Calculation"
        else:
            return "⚠️ Invalid input format for decay calculation."
    
    elif "dose" in query:
        dose = float(query.split()[1])
        result = radiation_protection_advice(dose)
        title = "Occupational Dose Assessment"
    elif "pridictedyeild" in query:
        activity = float(query.split()[1])
        result = predict_tc99m_yield(arrival_date, initial_activity)
        title = "Pridicted Yeild of Tc-99m"
    else:
        return "⚠️ Sorry, I didn't understand your request."
    
    # ✅ Format Response in Google Doc SNIF Style
    report = f"""
# **📊 Medical Physics (MP) Report**
### **📅 Date:** {current_date}
---

## **🔹 {title}**
{result}

---
**🔚 End of Report**
"""
    return report

#Tool-6 RP-Advice
@tool
def radiation_protection_advice(daily_exposure_microSv):
    """
    Function to calculate monthly radiation exposure and provide safety advice
    based on regulatory limits.
    
    Parameters:
    daily_exposure_microSv (float): Radiation exposure in microSieverts (µSv) for an 8-hour workday.

    Returns:
    str: Radiation safety advice.
    """
    # Constants
    WORK_DAYS_PER_MONTH = 22  # Assuming 22 working days in a month
    MONTHLY_LIMIT_mSv = OCCUPATIONAL_LIMITS["Monthly dose Limit to Worker"] # Monthly regulatory limit (20 mSv/year → 1.67 mSv/month)

    # Convert daily exposure from microSv to milliSv
    daily_exposure_mSv = daily_exposure_microSv / 1000  # 1 mSv = 1000 µSv

    # Calculate Monthly Exposure
    monthly_exposure_mSv = daily_exposure_mSv * WORK_DAYS_PER_MONTH

    # Provide safety advice based on exposure levels
    advice = f"Estimated Monthly Dose: {monthly_exposure_mSv:.3f} mSv\n"
    
    if monthly_exposure_mSv < MONTHLY_LIMIT_mSv:
        advice += "✅ Exposure is within the safe limits. Continue monitoring and follow ALARA principles."
    elif monthly_exposure_mSv < 2 * MONTHLY_LIMIT_mSv:
        advice += "⚠️ Exposure is approaching the limit. Reduce unnecessary exposure, optimize shielding, and monitor closely."
    else:
        advice += "🚨 Exposure exceeds safe limits! Immediate review required. Restrict exposure and implement strict radiation safety measures."

    return advice



# Tool6-Pridicted Yeild Tc-99m
@tool
def predict_tc99m_yield(arrival_date, initial_activity):
    """
    Predicts the daily yield of a Tc-99m generator for the next 7 days.
    
    Parameters:
    arrival_date (str): Date when the generator arrives (format: "DD-MM-YYYY").
    initial_activity (float): Initial activity of Mo-99 in mCi.
    
    Returns:
    dict: Dictionary with dates as keys and predicted yields as values.
    """
    # Constants
    #half_life_Mo99 = 66  # Mo-99 half-life in hours
    half_life_Mo99=RADIOACTIVE_NUCLIDES["Mollebdenium (MO-99)"]
    print(half_life_Mo99)
    decay_constant = math.log(2) / half_life_Mo99  # Decay constant
    extraction_efficiency = 0.87  # Typical Tc-99m extraction efficiency
    
    # Convert string date to datetime object
    arrival_datetime = datetime.strptime(arrival_date, "%d-%m-%Y")
    
    # Predict yield for the next 7 days
    predicted_yields = {}
    
    for day in range(1, 8):  # Next 7 days
        current_date = arrival_datetime + timedelta(days=day)
        time_elapsed = day * 24  # Convert days to hours
        remaining_activity = initial_activity * math.exp(-decay_constant * time_elapsed)
        tc99m_yield = remaining_activity * extraction_efficiency  # Apply efficiency
        predicted_yields[current_date.strftime("%d-%m-%Y")] = round(tc99m_yield, 2)
    
    return predicted_yields

tools=[radiation_protection_advice,predict_tc99m_yield,print_exposure_limits,patient_release_decision,unit_conversion_tool,radioactive_decay_tool]

#agent=initialize_agent(tools,llm,agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)
response=agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)

#response=agent.invoke({"input":" Neck dose of iodine treated patient is 20 microSv he is staying 24 hrs and his sef is BAD.  "})
#print(f"\n{response}\n")
import gradio as gr

# Define the function that handles user input
def process_input(user_input):
    # Assuming agent.invoke(user_input) is your processing logic
    # You should replace this with the actual call to your agent
    # For example: response = agent.invoke(user_input)
    
    # For now, we are just simulating the response with a simple message
    response = agent.invoke(user_input) # Replace with actual agent invocation
    return response


# Gradio UI
with gr.Blocks() as ui:
    
    # Inject inline CSS using Markdown (works properly in Gradio)
    gr.Markdown("""
        <style>
            body {
                background-color: black !important; /* Set background to black */
            }
            .gradio-container {
                background-color: black !important; /* Set container background */
                color: white !important; /* Default text color */
                font-family: Arial, sans-serif;
            }
            h1 {
                color: white !important; /* White heading color */
                text-align: center !important;
                font-weight: bold !important;
                font-size: 20px !important;
            }
            p {
                color: blue !important; /* Blue paragraph text */
                font-size: 16px !important;
                text-align: center !important;
                margin-top: 10px !important;
            }
            h3 {
                color: Blue !important; /* White heading color */
                text-align: right !important;
                font-weight: bold !important;
                font-size: 20px !important;
            }
        </style>
    """)

  
    # Display title and description
    gr.Markdown("<h1>AI-Powered MEDPHY-Nexus</h1>")#, unsafe_allow_html=True)
    gr.Markdown("<p>An intelligent system for unit conversions, decay calculations, the management of Iodine-treated patient releases, Pridicted yeild of Mo-99,ALARA advice to exposed worker, Ensures compliance with radiation protection regulations and medical physics standards, integrating all relevant safety limits and guidelines.</p>")#, unsafe_allow_html=True)
    
    # User input and output text
    user_input = gr.Textbox(label="Enter your Prompt")
    output_text = gr.Textbox(label="Response", interactive=False)
    
    # Submit button
    submit_button = gr.Button("Submit")
    
    # Define button interaction
    submit_button.click(fn=process_input, inputs=user_input, outputs=output_text)
    
    # Footer text
    gr.Markdown("<h3>Developed by Medical Physics Division(AEMCK)</h3>")#, unsafe_allow_html=True)

# Launch the Gradio app with public URL
ui.launch(share=True)
