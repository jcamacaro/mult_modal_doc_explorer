from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask import current_app as app
from bson.objectid import ObjectId
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain, ConversationChain
import pickle
from datetime import datetime
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
import time
import os


conversations_bp = Blueprint('conversations', __name__, template_folder='templates', url_prefix='/conversations')


@conversations_bp.route('/')
def conversations():
    conversations = list(app.mongo.db.conversations.find())
    return render_template('conversations.html', conversations=conversations)


@conversations_bp.route('/new', methods=['GET', 'POST'])
def new_conversation():
    topics = app.topics

    if request.method == 'POST':
        conversation_name = request.form.get('conversation_name')
        topic = request.form.get('topic')

        conversation = {
            'name': conversation_name,
            'topic': topic,
            'last_updated': datetime.utcnow()
        }
        result = app.mongo.db.conversations.insert_one(conversation)
        conversation_id = result.inserted_id

        return redirect(url_for('conversations.enter_conversation', conversation_id=conversation_id))

    return render_template('new_conversation.html', topics=topics)


@conversations_bp.route('/enter/<conversation_id>', methods=['GET', 'POST'])
def enter_conversation(conversation_id):
    postgres_user = os.getenv('DB_USER')
    postgres_password = os.getenv('DB_PASSWORD')
    postgres_db = os.getenv('DB_NAME')
    pgvector_host = os.getenv('DB_HOST')
    pgvector_port = os.getenv('DB_PORT')

    connection_string = f'postgresql://{postgres_user}:{postgres_password}@{pgvector_host}:{pgvector_port}/{postgres_db}'

    conversation = app.mongo.db.conversations.find_one({'_id': ObjectId(conversation_id)})
    if not conversation:
        os.abort(404)

    topic = conversation.get('topic', None)

    if request.method == 'POST':
        user_message = request.form.get('message')
        if user_message is not None:
            if topic != 'General (no documents)':
                # persist_directory = os.path.join(app.instance_path, f"db_{topic}")
                embeddings = app.embeddings #OpenAIEmbeddings()
                vectordb = PGVector(
                    connection_string=connection_string,
                    embedding_function=embeddings,
                    collection_name=topic
                )

                memory = ConversationBufferMemory(
                    memory_key='chat_history',
                    return_messages=True,
                    output_key='answer'
                )

                retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=app.the_llm,
                    memory=memory,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    get_chat_history=lambda h: h
                )


                chat_history = []
                llm_response = qa_chain({"question": user_message, "chat_history": chat_history})
                assistant_response = llm_response['answer']
                sources = []
                for source in llm_response["source_documents"]:
                    if source.metadata not in sources:
                        sources.append(source.metadata)

                if sources:  # Check if sources is not empty
                    sources_string = ', '.join(
                        source['source'].replace('app/uploads',
                                                 f'uploads/{topic}') for source in sources if 'source' in source)
                else:
                    sources_string = ""
                sources_list = sources_string.split(", ")
                # Update the conversation in MongoDB

            else:
                sources_list = []
                window_memory = ConversationBufferWindowMemory(k=3)
                qa_chain = ConversationChain(
                    llm=app.the_llm,
                    memory=window_memory,
                    verbose=False
                )
                llm_response = qa_chain({"input": user_message})
                assistant_response = llm_response['response']

            conversation_update = {
                "$push": {
                    "messages": {
                        "$each": [
                            {
                                "role": "user",
                                "content": user_message,
                                "timestamp": int(time.time() * 1000)
                            },
                            {
                                "role": "assistant",
                                "content": assistant_response,
                                "sources": sources_list,
                                "timestamp": int(time.time() * 1000)
                            }
                        ]
                    }
                }
            }
            app.mongo.db.conversations.update_one({"_id": ObjectId(conversation_id)}, conversation_update)
            return jsonify({"user_message": user_message,
                            "assistant_response": assistant_response,
                            "sources_list": sources_list})
        else:
            flash('User message is empty')

    conversation = app.mongo.db.conversations.find_one({"_id": ObjectId(conversation_id)})
    return render_template('enter_conversation.html', conversation=conversation, topic=topic)


@conversations_bp.route('/delete/<conversation_id>')
def delete_conversation(conversation_id):
    app.mongo.db.conversations.delete_one({'_id': ObjectId(conversation_id)})
    return redirect(url_for('conversations.conversations'))


def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
