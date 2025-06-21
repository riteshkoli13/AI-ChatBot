from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent
from browser_use import Agent, Browser, BrowserConfig
from dotenv import load_dotenv
import asyncio
import os


load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")  

browser = Browser(
    config=BrowserConfig(
       
        chrome_instance_path='C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',  
     
    )
)
initial_actions = [
	{'open_tab': {'url': 'https://www.flipkart.com/'}}
]
async def flipkart(product_details):
    
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key=api_key)
    
    agent = Agent(
        task=f"""
    
        Find product information on flipkart:
        - Product: {product_details}

         - Required details:
             1. Full product name (exact as shown on Amazon)
             2. Current minimum price
             3. Average rating
            4. Direct purchase URL
        - Format response as structured JSON data with these keys: "product_name", "price", "rating", "purchase_url"
        - If multiple sellers exist, return the lowest price option from a reputable seller
        -if particular product is not available on flipkart, then return a message "Product not available on flipkart"
        - Note: Return only factual information as displayed on the Amazon product page""",
        llm=llm,
        browser=browser,
        initial_actions=initial_actions
    )
    result = await agent.run()
    return result.final_result()


def get_flipkart_output(input):
    return asyncio.run(flipkart(input))


