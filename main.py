# Import necessary libraries
from crewai import Agent, Task, Crew, Process
import amazon
import flipkart
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent as BrowserAgent, Browser, BrowserConfig
from dotenv import load_dotenv
import asyncio
import os
import json
from langchain_groq import ChatGroq
# Load environment variables
load_dotenv()
#api_key = os.getenv("GOOGLE_API_KEY")

# Initialize browser for web scraping


os.environ["GROQ_API_KEY"] = "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"


# Initialize the Groq LLM
llm = ChatGroq(
    model_name="groq/llama3-8b-8192",
    temperature=0.7,
)



user_input="Wild Stone Edge EDP Premium Perfume for Men, 100 Ml"



product_name_extractor = Agent(
    role="Product Parser",
    goal="Extract product details from user input",
    backstory="""You analyze user queries to identify product names, quantities, and filters.
    """,
    verbose=True,
    allow_delegation=False,
    llm=llm
)




product_name_extractor_task = Task(
    description=f"""
                from the user input for product search details: "{user_input}"
               
                
                Extract:
                1. Product name
                2. Quantity/volume (if specified)
                3. Price constraints (if specified)
                4. Any other relevant filters
                

                example:
                input=Wild Stone Edge EDP Premium Perfume for Men, 100 Ml

                output=
                    product:Wild Stone Edge EDP Premium Perfume for Men,
                    quantity: 100 Ml
                
                
                
                """,
                
    expected_output=  """Return the results as a JSON object with these fields (include only if present in input):
                
                    "product": "extracted product name",
                    "quantity": "extracted quantity/volume",
                    "price_max": maximum price (number only),
                    "other_filters": "any other specifications"
                
                """,
                agent=product_name_extractor
            )
            

#{product_details}



#input={"user_input":"Wild Stone Edge EDP Premium Perfume for Men, 100 Ml"}

crew = Crew(agents=[product_name_extractor,], tasks=[product_name_extractor_task])
product_details = crew.kickoff()



amazon_product_details=amazon.get_amazon_output(product_details)
flipkart_product_details=flipkart.get_flipkart_output(product_details)

print(amazon_product_details)
print(flipkart_product_details)

response_generator_agent = Agent(
    role="Response Generator",
    goal="Create personalized, user-friendly responses",
    backstory="""You transform technical product information into friendly,
    helpful responses that highlight the most relevant information for the user.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)
response_generator_agent_task = Task(
    description=f"""
    Generate a short and user-friendly response for the product: {user_input}.

    best deal for this product on amazon:
    Amazon product details: {amazon_product_details}

    best deal for this product on flipkart:
    Flipkart product details: {flipkart_product_details}

    The response should:
    - Start with a warm greeting.
    - Mention the product name clearly.
    - Present the best deal info from both Amazon and Flipkart in the following format:

    Product: <Product Name>

    Amazon Deal:
    1. Full product name (as shown on Amazon)
    2. Current minimum price
    3. Average rating
    4. Direct purchase URL

    Flipkart Deal:
    1. Full product name (as shown on Flipkart)
    2. Current minimum price
    3. Average rating
    4. Direct purchase URL

    Additional instructions:
    - Use appropriate emojis in the final output to enhance user experience.
    - Keep the response concise and well-formatted.
    - Focus mainly on price and direct purchase links.
    """,
    expected_output="A short, friendly, emoji-enhanced response showing Amazon and Flipkart deals with a clear focus on pricing and buy links.",
    agent=response_generator_agent
)

crew = Crew(agents=[response_generator_agent,], tasks=[response_generator_agent_task])
product_details = crew.kickoff()