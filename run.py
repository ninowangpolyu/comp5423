import streamlit as st
from PIL import Image
import google.generativeai as genai
from model_interface import preprocess_our_model
import time
from transformers import BartForConditionalGeneration, BertTokenizer, BartConfig
import torch


# Streamed response emulator

def response_generator(response):
    for word in response.split():
        yield word
        time.sleep(0.05)


@st.cache_resource
def init_model(status_model):
    print(status_model)
    model_path = './model/model.pt'
    output_dir = './model'
    print("Loading model...")
    print("Model path:", model_path)
    model = torch.load(model_path, map_location=torch.device('cpu'))
    print("Model loaded.")
    tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=False)
    print("Tokenizer loaded.")
    return model, tokenizer


st.set_page_config(page_title="Gemini Pro with Streamlit", page_icon="♊")

status_model = "init model"
model, tokenizer = init_model(status_model)

st.write("欢迎来到 Gemini Pro 聊天机器人。您可以通过提供您的 Google API 密钥来继续。")

with st.expander("提供您的 Google API 密钥"):
    google_api_key = st.text_input("Google API 密钥", key="google_api_key", type="password")

google_api_key = 'AIzaSyA5AcrN9fAW-PrGzPRG3kEfX1OftWssT3k'
if not google_api_key:
    st.info("请输入 Google API 密钥以继续")
    st.stop()

genai.configure(api_key=google_api_key)

st.title("Gemini Pro 与 Streamlit 聊天机器人")

with st.sidebar:
    option = st.selectbox('选择您的模型', ('gemini-pro', 'gemini-pro-vision','our model'))

    if 'model' not in st.session_state or st.session_state.model != option:
        st.session_state.chat = genai.GenerativeModel(option).start_chat(history=[])
        st.session_state.model = option

    st.write("在此处调整您的参数:")
    temperature = st.number_input("温度", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    max_token = st.number_input("最大输出令牌数", min_value=0, value=1024)
    gen_config = genai.types.GenerationConfig(max_output_tokens=max_token, temperature=temperature)

    st.divider()

    upload_image = st.file_uploader("在此上传您的图片", accept_multiple_files=False, type=['jpg', 'png'])

    if upload_image:
        image = Image.open(upload_image)
    st.divider()

    if st.button("清除聊天历史"):
        st.session_state.messages.clear()
        st.session_state["messages"] = []

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if upload_image:
    if option == "gemini-pro":
        st.info("请切换到 Gemini Pro Vision")
        st.stop()
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = st.session_state.chat.send_message([prompt, image], stream=True, generation_config=gen_config)
        response.resolve()
        msg = response.text

        st.session_state.chat = genai.GenerativeModel(option).start_chat(history=[])
        st.session_state.messages.append({"role": "assistant", "content": msg})

        st.image(image, width=300)
        st.chat_message("assistant").write(msg)

else:
    if option == "gemini-pro":
        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            response = st.session_state.chat.send_message(prompt, stream=True, generation_config=gen_config)
            response.resolve()
            msg = response.text
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
    else:
        if prompt := st.chat_input():
            st.chat_message("user").write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            input_all = []
            for m in st.session_state.messages:
                input_all.append(m['content'])
            print(input_all)
            response = preprocess_our_model(input_all, tokenizer, model)
            with st.chat_message("assistant"):
                response = st.write_stream(response_generator(response))
            st.session_state.messages.append({"role": "assistant", "content": response})
