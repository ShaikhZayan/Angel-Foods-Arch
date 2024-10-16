from fastapi import FastAPI, Depends, HTTPException, Header, Request, File, Form, UploadFile
from typing import Annotated
from jose import JWTError
from datetime import timedelta
import uvicorn
from utils import create_access_token, create_decode_token
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
import json
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_core.prompts import PromptTemplate
import string
import pandas as pd

app = FastAPI()


# Load environment variables
load_dotenv()

# Set up the environment for the API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDZXndpuiEZJB8_0j-7Yl1IcsxDimigQdM"

password = "zayan"
username = "zayan"

fake_users_db: dict[str, dict[str, str]] = {
    "zayan": {
        "username": username,
        "full_name": "shaikh zayan",
        "email": "siddiquizayan50@gmail.com",
        "password": password
    }

}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Add CORS middleware
allowed_origins = ["*"]  # Change in production to restrict access
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type"],
)






# @app.post("/main-model")
# async def model_new(query: str):
#     # Define API key and LLM
#     # groq_api = "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"
#     # llm = ChatGroq(temperature=0.4, model="llama3-70b-8192", api_key=groq_api, verbose=True)
    
#     # Initialize the Google LLM (Gemini)
#     llm = ChatGoogleGenerativeAI(
#         model='gemini-1.5-flash',
#         verbose=True,
#         temperature=0.2,
#         safety_settings={
#             HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
#             HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
#             HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
#             HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#             },
#         google_api_key=os.getenv('GOOGLE_API_KEY')
#     )



#     # Use the create_csv_agent function to initialize the tool
#     tool = create_csv_agent(llm, "Angel.csv", verbose=True, allow_dangerous_code=False)

#     def query_data(query):
#         # Instructions for the LLM
#         instructions = """
#         You are an expert data analyst using the latest AI technologies. 
#         Your job is to analyze the data from the CSV file and provide concise, clear answers without any unnecessary repetition.
#         Please ensure that your responses are clear, accurate, and free of additional or irrelevant information.
#         Avoid repeating phrases, and ensure that your answer concludes with only one complete sentence if possible.

#         If your answer includes any additional statements or information after a complete sentence, please omit those extras. 
#         If the query is unclear, ask for clarification before proceeding.
    
#         Let me give you some specific instructions about our CSV data:
        
#         - The data consists of 16 columns:
#           1. customer_reference
#           2. company_name
#           3. Banner
#           4. Cluster
#           5. Cluster Size
#           6. suburb
#           7. postalcode
#           8. state
#           9. value
#           10. cash_on_delivery
#           11. code
#           12. description
#           13. uom_qty
#           14. qty
#           15. unit_of_Measure
        
#         - Special Handling:
#           - The **customer_reference** column contains order numbers. Some order numbers repeat, so **do not calculate** repeated order numbers.
#           - The **company_name** is the name of the company or store, along with the short area or street address. The **Banner** is just the company name without the street address.
#           - The **value** column represents the total order value (price).
#           - The **cash_on_delivery** is typically 'No', meaning most orders are not delivered with cash on delivery.
#           - The **qty** (Quantity) represents the total number of units in the order. If the **qty** is greater than 8, this indicates at least **1 carton**. 
#           - The **uom_qty** (Unit Order Measurement Quantity) reflects the number of cartons:
#             - 1 carton = 8 units. 
#             - For example, if **qty** is 16, there are **2 cartons**, so **uom_qty** should be 2.
    
#         Please use these instructions when analyzing the data and answering the query.
#         """
        
#         # Combine instructions with the user query
#         full_query = f"{instructions}\n\nUser Query: {query}"
        
#         # Run the query with the tool
#         response = tool.run(full_query)
#         return response

#     response = query_data(query)


#     # Print the beautified response
#     return {"result": response}

   









@app.post("/main-model")
async def model_new(query: str):
    # Define API key and LLM
    # groq_api = "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"
    # llm = ChatGroq(temperature=0.4, model="llama3-70b-8192", api_key=groq_api, verbose=True)
    
    # Initialize the Google LLM (Gemini)
    llm = ChatGoogleGenerativeAI(
        model='gemini-1.5-flash',
        verbose=True,
        temperature=0.2,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            },
        google_api_key=os.getenv('GOOGLE_API_KEY')
    )

    df = pd.readcsv()

    # Use the create_csv_agent function to initialize the tool
    tool = create_csv_agent(llm, "Angel.csv", verbose=True, allow_dangerous_code=False)

    def query_data(query):
        # Instructions for the LLM
        instructions = """
        You are an expert data analyst using the latest AI technologies. 
        Your job is to analyze the data from the CSV file and provide concise, clear answers without any unnecessary repetition.
        Please ensure that your responses are clear, accurate, and free of additional or irrelevant information.
        Avoid repeating phrases, and ensure that your answer concludes with only one complete sentence if possible.

        If your answer includes any additional statements or information after a complete sentence, please omit those extras. 
        If the query is unclear, ask for clarification before proceeding.
    
        Let me give you some specific instructions about our CSV data:
        
        - The data consists of 16 columns:
          1. customer_reference
          2. company_name
          3. Banner
          4. Cluster
          5. Cluster Size
          6. suburb
          7. postalcode
          8. state
          9. value
          10. cash_on_delivery
          11. code
          12. description
          13. uom_qty
          14. qty
          15. unit_of_Measure
        
        - Special Handling:
          - The **customer_reference** column contains order numbers. Some order numbers repeat, so **do not calculate** repeated order numbers.
          - The **company_name** is the name of the company or store, along with the short area or street address. The **Banner** is just the company name without the street address.
          - The **value** column represents the total order value (price).
          - The **cash_on_delivery** is typically 'No', meaning most orders are not delivered with cash on delivery.
          - The **qty** (Quantity) represents the total number of units in the order. If the **qty** is greater than 8, this indicates at least **1 carton**. 
          - The **uom_qty** (Unit Order Measurement Quantity) reflects the number of cartons:
            - 1 carton = 8 units. 
            - For example, if **qty** is 16, there are **2 cartons**, so **uom_qty** should be 2.
    
        Please use these instructions when analyzing the data and answering the query.
        """
        
        # Combine instructions with the user query
        full_query = f"{instructions}\n\nUser Query: {query}"
        
        # Run the query with the tool
        response = tool.run(full_query)
        return response

    response = query_data(query)


    # Print the beautified response
    return {"result": response}










   
   
   
   
   

@app.get("/")
async def root():
    return {"Welcome To Angle Foods API"}

@app.post("/login")
def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends(OAuth2PasswordRequestForm)]
):
    """
    Understanding the login system
    -> Takes form_data that have username and password
    """
    
    user_in_fake_db = fake_users_db.get(form_data.username)
    if not user_in_fake_db: 
        raise HTTPException(status_code=400, detail="Incorrect Username")   
    
    if not form_data.password == user_in_fake_db["password"]:
        raise HTTPException(status_code=400, detail="Incorrect Password")
    

    access_token = create_access_token(subject=user_in_fake_db["username"])

    return {"access_token": access_token, "token_type": "bearer"}



@app.get("/get-access-token")
def get_access_token(token: Annotated[str, Depends(oauth2_scheme)]):
    """
    Understanding the access token
    -> Takes user_name as input and returns access token
    -> timedelta(minutes=1) is used to set the expiry time of the access token to 1 minute
    """  
    # Validate the token and get the user from the decoded token data
    try:
        decoded_token_data = create_decode_token(token)
        user_in_db = fake_users_db.get(decoded_token_data["sub"])
        
        if user_in_db is None:
            raise HTTPException(status_code=401, detail="User not found")

        # Only if the token is valid and the user exists, generate a new token
        access_token = create_access_token(
            subject=user_in_db["username"]
        )
    
        return {"access_token": access_token, "token_type": "bearer"}

    
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")



@app.get("/header")
def getheader(request: Request):
    headers = request.headers
    
    return dict(headers)



@app.get("/decode_token")
def decoding_token(token: Annotated[str, Depends(oauth2_scheme)]):
    """ 
    Understanding the access token decoding and validation
    """
    try:
        decoded_token_data = create_decode_token(token)
        return {"decoded":decoded_token_data}
    except JWTError as e:
        raise HTTPException(status_code=401, detail="Invalid token")



if __name__ == "__main__":
    uvicorn.run('main:app', host = '0.0.0.0', port = '8001', reload = True)
