"""
E-Commerce Shopping AI Assistant
Author: Prabhakar Gadupudi
# I am running this script on my Ubutnu
# Python 3.12.3
# Description:    Ubuntu 24.04.4 LTS
# Release:        24.04
# Codename:       noble

"""

import json
import re
import pandas as pd
from ollama import chat



# LLM Wrapper
class LLMWrapper:

    def __init__(self, model="qwen2.5:7b"):
        # Save the default model name for future requests.
        self.model = model

    def ask(self, prompt):
        # Query the Ollama chat API and return the assistant's content.
        response = chat(model=self.model, messages=[{"role": "user", "content": prompt}])

        return response["message"]["content"]

# JSON Extractor
def extract_json(text):

    # Trim whitespace around the response text.
    text = text.strip()

    # Remove markdown fences if present in the assistant's output.
    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.replace("```", "")

    # Find the first JSON object in the text block.
    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if not match:
        raise ValueError(
            f"No JSON found in:\n{text}"
        )

    # Parse and return the JSON payload.
    return json.loads(match.group())



# Requirement AI Agent
class RequirementAIAgent:

    def __init__(self, llm):
        # Store the LLM wrapper used to generate requirement extraction.
        self.llm = llm

    def process(self, query):

        prompt = f"""
You are a shopping requirement extractor.

Return ONLY valid JSON.

No markdown.
No explanation.
No code fences.

Schema:

{{
  "category": "",
  "budget": null,
  "purpose": [],
  "sort": "",
  "brand": ""
}}

Examples:

Query:
show me gaming laptops under 80k

Output:
{{
  "category":"Laptop",
  "budget":80000,
  "purpose":["gaming"],
  "sort":"",
  "brand":""
}}

Query:
show me all laptops

Output:
{{
  "category":"Laptop",
  "budget":null,
  "purpose":[],
  "sort":"",
  "brand":""
}}

Query:
show me cheap laptops

Output:
{{
  "category":"Laptop",
  "budget":null,
  "purpose":[],
  "sort":"price_asc",
  "brand":""
}}

User Query:

{query}
"""

        response = self.llm.ask(prompt)

        print("\n========== RequirementAIAgent ==========")
        print(response)

        return extract_json(response)



# Product Search Tool
class ProductSearchTool:

    def search(self, requirements):

        # Load the product catalog from a CSV file.
        df = pd.read_csv("products.csv")

        category = requirements.get("category")

        if category:
            # Filter products by category if specified.
            df = df[df["category"].str.lower() == category.lower()]

        budget = requirements.get("budget")

        if budget:
            # Keep only products within the user's budget.
            df = df[df["price"] <= budget]

        brand = requirements.get("brand")

        if brand:
            # Filter by brand name when provided.
            df = df[df["brand"].str.lower() == brand.lower()]

        purposes = requirements.get("purpose", [])

        if purposes:
            # Match product tags against target purchase purposes.
            mask = []

            for _, row in df.iterrows():
                tags = str(row["tags"]).lower()
                matched = any(p.lower() in tags for p in purposes)
                mask.append(matched)

            df = df[mask]

        sort = requirements.get("sort")

        if sort == "price_asc":
            # Sort results by ascending price.
            df = df.sort_values(by="price", ascending=True)
        elif sort == "price_desc":
            # Sort results by descending price.
            df = df.sort_values(by="price", ascending=False)

        return df



# Recommendation AI Agent
class RecommendationAIAgent:

    def __init__(self, llm):
        # Use the same LLM wrapper for recommendation generation.
        self.llm = llm

    def process(self, user_query, requirements, products):
        if products.empty:
            # Return early if no matches were found.
            return "No matching products found."

        # Convert product rows into JSON format for the LLM prompt.
        products_json = products.to_dict(orient="records")

        prompt = f"""
You are a shopping advisor.

User Query:

{user_query}

Requirements:

{json.dumps(requirements, indent=2)}

Products:

{json.dumps(products_json, indent=2)}

Tasks:

1. Recommend best products.
2. Explain why.
3. Compare products.
4. Mention pros and cons.
5. Keep response concise.
"""

        # Ask the model to provide a product recommendation summary.
        return self.llm.ask(prompt)



# Shopping Manager AI Agent
class ShoppingManagerAIAgent:

    def __init__(self):
        # Set up the assistant components: LLM wrapper, requirement agent, search tool, and recommendation agent.
        self.llm = LLMWrapper()
        self.requirement_agent = RequirementAIAgent(self.llm)
        self.search_tool = ProductSearchTool()
        self.recommendation_agent = RecommendationAIAgent(self.llm)

    def execute(self, query):
        # Convert the raw user query into structured search requirements
        requirements = self.requirement_agent.process(query)
        print("\nExtracted Requirements:")
        print(json.dumps(requirements, indent=2))

        # Search the product catalog based on extracted requirements
        products = self.search_tool.search(requirements)
        print(f"\nProducts Found: {len(products)}")

        # Generate the final recommendation text
        recommendation = self.recommendation_agent.process(query, requirements, products)
        return recommendation


if __name__ == "__main__":
    # Example queries
    questions = [
        "I want to buy a laptop under 80k for programming and gaming",
        "show me all laptops order by descending order",
        "show me cheap laptops",
        "show me laptops for programming",
        "show me gaming laptops",
        "show me top 2 gaming laptops",
        "show me gaming laptops order by price descending",
        "Show me mobiles with 12GB RAM",
        "Show me 55 inch OLED TVs",
        "Show me educational toys for kids",
        "Show me products with more than 20% discount",
        "Show me gift cards under 1000",
        "Show me headphones with noise cancellation",
        "Show me top rated products across all categories"
    ]
    print ("_"*100)
    print ("Here are example questions")
    for idx, q in enumerate(questions):
        print (f"{idx+1}. {q}")
    print ("_"*100)


    query = input("\nAsk Shopping Assistant: ")

    manager = ShoppingManagerAIAgent()

    result = manager.execute(query)

    print("\n")
    print("=" * 80)
    print("FINAL RESPONSE")
    print("=" * 80)
    print(result)
    print("=" * 80)
