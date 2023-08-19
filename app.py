import streamlit as st
from PyPDF2 import PdfReader
import time
import openai
import pickle
import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS 
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from api_keys import openai_api


# from streamlit_extras.add_vertical_space import add_vertical_space
import streamlit as st

# Create a string to hold the HTML for the title
# Create a string to hold the HTML for the title
# Create a string to hold the HTML for the title

title_html = """
<div class="title-container">
    <h1 class="main-title">
        <span style="animation-delay: 0s;">S</span>
        <span style="animation-delay: 1s;">K</span>
        <span style="animation-delay: 2s;">I</span>
        <span style="animation-delay: 2s;">L</span>
        <span style="animation-delay: 3s;">L</span>
        <span style="animation-delay: 4s;">   </span>
        <span style="animation-delay: 5s;">   </span>
        <span style="animation-delay: 6s;">S</span>
        <span style="animation-delay: 7s;">E</span>
        <span style="animation-delay: 8s;">T</span>
        <span style="animation-delay: 9s;">   </span>
        <span style="animation-delay: 10s;">   </span>
        <span style="animation-delay: 11s;">B</span>
        <span style="animation-delay: 12s;">E</span>
        <span style="animation-delay: 13s;">A</span>
        <span style="animation-delay: 14s;">C</span>
        <span style="animation-delay: 15s;">O</span>
        <span style="animation-delay: 16s;">N</span>       
    </h1>
</div>
"""

# Add CSS styles for the typing animation
animation_css = """
<style>
    .title-container {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        font-family: 'Algerian', sans-serif; /* Apply Algerian font */
    }
    .main-title {
        color: #ffff99;
        font-size: 36px;
        font-weight: 700;
        margin: 0;
        white-space: nowrap;
        overflow: hidden;
        border-right: 2px solid transparent;
        animation: typing 5s steps(100) forwards, blink-caret 0.75s step-end infinite;
    }
    @keyframes typing {
        from {
            width: 0;
        }
        to {
            width: 100%;
        }
    }
    @keyframes blink-caret {
        from,
        to {
            border-color: transparent;
        }
        50% {
            border-color: #003366;
        }
    }
    .sub-title {
        color: #009688;
        font-size: 24px;
        margin: 0;
    }
    .main-title.finished {
        animation: none;
    }
    .main-title.finished::after {
        content: "";
        display: block;
        width: 100%;
        height: 2px;
        background-color: #003366;
        position: relative;
        animation: blink-caret 0.75s step-end infinite;
    }
    @keyframes animationfinished {
        to {
            width: 100%;
        }
    }
</style>
"""

# Combine the CSS styles and title HTML into a single string
combined_html = f"{animation_css}{title_html}"

# Display the combined HTML using Streamlit's "st.markdown" function with unsafe_allow_html=True
st.markdown(combined_html, unsafe_allow_html=True)



# List of image URLs
image_urls = [
    'https://cdn-icons-png.flaticon.com/128/4727/4727496.png',
    'https://cdn-icons-png.flaticon.com/128/2631/2631384.png',
    'https://cdn-icons-png.flaticon.com/128/10484/10484260.png',
    'https://cdn-icons-png.flaticon.com/128/3775/3775431.png',
    # 'https://cdn-icons-png.flaticon.com/512/1158/1158432.png',
    'https://cdn-icons-png.flaticon.com/128/11112/11112991.png',
    # Add more image URLs as needed
]
# Create a list of image HTML elements
images_html = ""

# Build the HTML for each image with a hover effect
for url in image_urls:
    images_html += f'<img src="{url}" class="hover-image" style="max-width: 300px; margin: 10px;">'

# Add CSS styles for the hover effect
hover_effect_css = """
<style>
    .hover-image {
        transition: transform 0.3s; /* Add a smooth transition effect */
    }
    .hover-image:hover {
        transform: scale(1.1); /* Enlarge the image on hover */
    }
</style>
"""

# Combine the CSS styles and images HTML into a single string
combined_html = f"{hover_effect_css}<div style='display: flex;'>{images_html}</div>"

# Display the combined HTML using Streamlit's "st.markdown" function with unsafe_allow_html=True
st.markdown(combined_html, unsafe_allow_html=True)
# st.markdown(images_row_html, unsafe_allow_html=True)






subtitle="""<div>
<h2 style="color: #009688; font-size: 24px; margin: 0;">Upload Your Resume in pdf format</h2>
</div>"""
st.markdown(subtitle,unsafe_allow_html=True)

API=openai_api
load_dotenv()

def main():
    llm=OpenAI(openai_api_key=API,
           temperature=0.7)
    # st.header("Upload Your Resume in pdf format")
    pdf=st.file_uploader("upload",type="pdf")

    if pdf is not None:
        doc_reader=PdfReader(pdf)
        st.write("Name:",pdf.name)
        raw_text=""
        for i,page in enumerate(doc_reader.pages):
            text=page.extract_text()
            if text:
                raw_text+=text

        
        text_splitter=CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=200,
            length_function=len)
        texts=text_splitter.split_text(raw_text)
        
        file_name=pdf.name.replace(".pdf","rb")
        if os.path.exists(f"{file_name}.pkl"):
            with open(f"{file_name}.pkl","rb") as f:
                docsearch=pickle.load(f)
            # st.write("Embeddings Loaded from disk")
        else:
            embeddings=OpenAIEmbeddings(openai_api_key=API)
            docsearch=FAISS.from_texts(texts,embeddings)
            with open(f"{file_name}.pkl","wb") as f:
                pickle.dump(docsearch,f)
            # st.write("New Embeddings Created")

        chain = load_qa_chain(llm=llm,chain_type='stuff')
        # chain.llm_chain.prompt.template
        
        time.sleep(5)
        query="What is the name of candidate as per the resume?Only return the name"
        docs = docsearch.similarity_search(query)
        name=chain.run(input_documents=docs,question=query)
        st.write("Hello "+name+" Your resume is uploaded and being processed ")

        time.sleep(5)
        query="What are the skills of candidate as per the resume?Only return the skills"
        docs = docsearch.similarity_search(query)
        skills=chain.run(input_documents=docs,question=query)
        
        time.sleep(5)
        query="What are the skills of the candidate as per the resume?"
        docs = docsearch.similarity_search(query)
        skills=chain.run(input_documents=docs,question=query)
        print("SKILLS :",skills)
        
        time.sleep(7)
        query="What does the candidate aspire to become as per the resume?"
        docs = docsearch.similarity_search(query)
        job_role=chain.run(input_documents=docs,question=query)
        st.write("Applying for Job Role: ",job_role )
        
        time.sleep(7)
        query="As per the"+skills+"is the candidate suitable for becoming a "+job_role+"?"
        docs = docsearch.similarity_search(query)
        job_gurantee=chain.run(input_documents=docs,question=query)
        st.write(job_gurantee)
        
        time.sleep(7)
        query="As per the"+skills+"please tell him 5-6 tips to become a better "+job_role+"in bulleted format"
        docs = docsearch.similarity_search(query)
        improvements=chain.run(input_documents=docs,question=query)
        st.write("Improvements",improvements)


        llm2 = OpenAI(
                model_name="text-davinci-003", # default model
                openai_api_key=API,
                temperature=0.0)

        

        time.sleep(4)
        query=f"The {improvements} that you have suggested are great please tell the resources from you-tube/google for leearning those. "
        prompt = PromptTemplate(
            input_variables=["improvements"],
            template="The {improvements} that you have suggested are great please tell the resources from you-tube/google for leearning those. Make Sure to add the links.",
        )
        llmchain = LLMChain(llm=llm2, prompt=prompt)
        resources = llmchain.run(improvements)
        st.write(resources)


        time.sleep(4)
        prompt = PromptTemplate(
       input_variables=["skills", "job_role", "improvements"],
        template="Based on {skills} how suitable is the candidate for {job_role}.Rate out of 100. Also tell how suitable he will be once he has done these {improvements},rate out of 100.Give the intital score without improvements lower than after improvements and do not give anyone scores 100/100",
)
        llmchain = LLMChain(llm=llm2, prompt=prompt)
        scores = llmchain.run({"skills":skills, "job_role":skills,"improvements":improvements})
        st.write(scores)

        
        
        

if __name__ == '__main__':
    main()




