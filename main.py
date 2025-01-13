from dotenv import load_dotenv
import os

from langchain import PromptTemplate
from langchain.chains import LLMChain
load_dotenv()
api_key=os.getenv("GOOGLE_API_KEY")
prompt=input("input: ")
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
country=input("input: ")
#print(ai_msg.content)
prompt_template=PromptTemplate(
  input_variables=["country"],
  template="provide a good name for {country} cuisine"
)
chain=LLMChain(llm=llm,prompt=prompt_template)
result=chain.run(country)
print(result)