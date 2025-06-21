from crewai import Agent, Task, Crew, Process
import os
import json
from langchain_groq import ChatGroq
# Import the amazon and flipkart modules
import amazon
import flipkart
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Groq LLM with environment variable
def initialize_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        os.environ["GROQ_API_KEY"] = "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"
    
    return ChatGroq(
        model_name="groq/llama3-8b-8192",
        temperature=0.7,
    )

def extract_product_details(user_input, llm):
    """Extract product details from user input using CrewAI"""
    product_name_extractor = Agent(
        role="Product Parser",
        goal="Extract product details from user input",
        backstory="""You analyze user queries to identify product names, quantities, and filters.""",
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
            
        expected_output="""Return the results as a JSON object with these fields (include only if present in input):
            
                "product": "extracted product name",
                "quantity": "extracted quantity/volume",
                "price_max": maximum price (number only),
                "other_filters": "any other specifications"
            """,
        agent=product_name_extractor
    )
    
    # Create and run the crew
    crew = Crew(agents=[product_name_extractor], tasks=[product_name_extractor_task])
    return crew.kickoff()

def get_amazon_details(product_details):
    """Fetch product details from Amazon using the amazon module"""
    try:
        # Call the amazon module's function to get real data
        amazon_data = amazon.get_amazon_output(product_details)
        
        # If the result is a string (likely JSON), parse it
        if isinstance(amazon_data, str):
            try:
                amazon_data = json.loads(amazon_data)
            except:
                pass
        
        return amazon_data
    except Exception as e:
        print(f"Error getting Amazon details: {e}")
        # Fallback response if there's an error
        return {
            "product_name": "Error retrieving product",
            "price": "N/A",
            "rating": "N/A",
            "purchase_url": "https://www.amazon.in"
        }

def get_flipkart_details(product_details):
    """Fetch product details from Flipkart using the flipkart module"""
    try:
        # Call the flipkart module's function to get real data
        flipkart_data = flipkart.get_flipkart_output(product_details)
        
        # If the result is a string (likely JSON), parse it
        if isinstance(flipkart_data, str):
            try:
                flipkart_data = json.loads(flipkart_data)
            except:
                pass
        
        return flipkart_data
    except Exception as e:
        print(f"Error getting Flipkart details: {e}")
        # Fallback response if there's an error
        return {
            "product_name": "Error retrieving product",
            "price": "N/A",
            "rating": "N/A",
            "purchase_url": "https://www.flipkart.com"
        }

def generate_response(user_input, product_details, amazon_details, flipkart_details, llm):
    """Generate a user-friendly response with CrewAI"""
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
        Amazon product details: {amazon_details}
        
        best deal for this product on flipkart:
        Flipkart product details: {flipkart_details}
        
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
    
    # Create and run the crew
    crew = Crew(agents=[response_generator_agent], tasks=[response_generator_agent_task])
    result = crew.kickoff()
    
    # Return the raw output from the Response Generator agent
    return result

# Main function to process user input
def process_user_message(user_input):
    """Process user input and generate AI response"""
    try:
        # Initialize LLM
        llm = initialize_llm()
        
        # Extract product details
        product_details = extract_product_details(user_input, llm)
        
        # Get product details from Amazon and Flipkart
        amazon_details = get_amazon_details(product_details)
        flipkart_details = get_flipkart_details(product_details)
        
        # Generate response - this is directly passed to the user
        response = generate_response(user_input, product_details, amazon_details, flipkart_details, llm)
        
        # Return the raw response from the Response Generator agent
        return response
    except Exception as e:
        print(f"Error processing message: {e}")
        return f"I'm sorry, I couldn't process your request due to an error: {str(e)}"

# Helper function to integrate with Flask app.py
def process_user_input(user_input):
    """Process user input from Flask app"""
    try:
        # Get the Response Generator's output
        response = process_user_message(user_input)
        
        # Convert to string if needed
        if hasattr(response, 'raw'):
            return response.raw
        elif not isinstance(response, str):
            return str(response)
        return response
    except Exception as e:
        print(f"Error in process_user_input: {e}")
        return "I'm sorry, I encountered an error while processing your request. Please try again with a product search."