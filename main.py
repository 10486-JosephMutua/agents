from dotenv import load_dotenv
import os
from langchain.chains import SequentialChain
from re import template
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
load_dotenv()
api_key=os.getenv("GOOGLE_API_KEY")
country=input("input: ")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

prompt_templateone="""
provide a good name for{country} cuisine
"""
prompt_template1=PromptTemplate(
  input_variables=["country"],
  template=prompt_templateone
)

first_chain = LLMChain(llm=llm, prompt=prompt_template1,
                     output_key="country_name"
                    )

prompt_templatetwo="""
provide a menu for the {country_name}
"""
prompt_template2=PromptTemplate(
  input_variables=["country_name"],
  template=prompt_templatetwo
)
second_chain=LLMChain(llm=llm, prompt=prompt_template2,
                     output_key="list_menu"
                    )

prompt_templatethree="""
provide a price for  {list_menu}.make sure to list the prices on the right side of {list_menu}
"""
prompt_template3=PromptTemplate(
input_variables=["country_name"],
template=prompt_templatethree
)

third_chain = LLMChain(llm=llm, prompt=prompt_template3,
                     output_key="prices"

                    )
final_chain=SequentialChain(
    chains=[first_chain,second_chain,third_chain ],
    input_variables=["country"],
    output_variables=["country_name","list_menu","prices"],
    verbose=False
)
result=final_chain({"country":country})

print(result)
