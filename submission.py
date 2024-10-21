from flask import Flask, request, jsonify
from langchain_openai import OpenAIEmbeddings  # Updated import as per deprecation warning
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup
import os


if "USER_AGENT" not in os.environ:
    os.environ["USER_AGENT"] = "MyCustomBot/1.0 ( https://brainlox.com/courses/category/technical; manaspatil281@gmail.com)"


headers = {
    "User-Agent": os.environ["USER_AGENT"]
}

app = Flask(__name__)



def extract_courses(url):

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')


    courses = []
    for course_card in soup.find_all('div', class_="course-card"):
        title = course_card.find('h3').text
        description = course_card.find('p').text
        courses.append({"title": title, "description": description})

    return courses



url = "https://brainlox.com/courses/category/technical"
course_data = extract_courses(url)


embedding_model = OpenAIEmbeddings(openai_api_key="sk-JPWxMvdR9tQM6-wVbpxDCgk0W4BvL3ankAWvF4JDDzT3BlbkFJgw9Kzmq-P79Yh8sZUwmlgEaX__BpNg6NK5kKLFjlcA")


index_path = "faiss_vector_store"
index_file = os.path.join(index_path, "index.faiss")


vector_store = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)


@app.route('/chat', methods=['POST'])
def chat():

    user_input = request.json.get('message')


    user_embedding = embedding_model.embed_query(user_input)


    docs = vector_store.similarity_search(user_input, k=1)


    if docs:
        response = docs[0].page_content
    else:
        response = "No relevant course found."

    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(debug=True)
