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
#load 
load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key="AIzaSyBhYBqdvs0LyQvXQ_o-27BSsZnF7hOZEJ8")
