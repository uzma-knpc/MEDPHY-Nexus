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
