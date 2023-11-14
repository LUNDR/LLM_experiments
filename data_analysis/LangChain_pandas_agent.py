import os
from dotenv import load_dotenv

from langchain.agents import create_csv_agent, create_pandas_dataframe_agent
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import pandas as pd
# from langchain import PromptTemplate
from langchain.llms import OpenAI


load_dotenv()

OPENAI_MODEL = "gpt-4-1106-preview"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def main():
    file_path = "C:/Users/lundr/OneDrive/LangChain/data/API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_5728866.csv"
    data = pd.read_csv(file_path, skiprows=4)
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0),data, verbose=True )
    agent.agent.llm_chain.prompt.template
    prompt = "Execute the following steps in order:\
                       1. Subset the dataframe to include only data for Switzerland. \
                       2. identify any fields which do not contain numeric values \
                       3. drop those fields any any fields which do not have a field name.\
                       4. Transpose the dataframe so that remaining field names become the index. If there is a length mismatch add additional empty rows\
                       5. Convert the index to years.\
                       6. Plot a chart of the GDP per capita of Switzerland over time. Make the plot in XKCD style. \
                       7. Add a title to the plot of 'GDP per Capita $PPP: Switzerland'. \
                       8. Use a comma seperator for thousands on the y-axis.' \
                       9. Remove the legend\
                       10. save the plot as switzerland_gdp_per_capita_xkcd4.png. \
                       11. Print the first 10 rows of the dataframe used to plot the chart"
    
    prompt2 = "create a table of average GDP per Capita Growth for each decade for Switzerland and the UK, save this table as an image file 'table.png'"
    result = agent.run(prompt)
    result2 = agent.run(prompt2)
 
    print(result)
    print(result2)

if __name__=="__main__":
    main()
