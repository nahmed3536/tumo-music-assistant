import streamlit as st
from openai import OpenAI
from audiocraft.models import musicgen
import torch
import time
import numpy as np
import os


title = "SynthScribe"
title_bot = "Music Chatbot"
title_musicgen = "Audio Generator"

if torch.cuda.is_available():
    # Set the device to CUDA
    device = torch.device("cuda")
else:
    # Set the device to CPU
    device = torch.device("cpu")

if "model" not in st.session_state.keys():
    st.session_state['model'] = musicgen.MusicGen.get_pretrained('medium', device=device)

if "client" not in st.session_state.keys():
  st.session_state['client'] = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],
  )





### AI ASSISTANT CODE ###
version = "0.3"

def text_to_audio(prompt, duration):
    st.session_state.model.set_generation_params(duration=duration)
    sample = st.session_state.model.generate([prompt], progress=True)
    assert sample.dim() == 3
    sample = sample.detach().cpu()
    sample = sample.reshape(sample.shape[-1]).numpy()
    return sample


def chat_gpt(prompt,context):
    response = st.session_state.client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=context
    )
    answer =response.choices[0].message.content.strip()
    return answer

def assistant(prompt: str,context) -> str:
    return chat_gpt(prompt,context)



option = st.selectbox(
   "Select Option",
   ("Music Helper Chatbot", "Audio Generator", "Info"),
   index=None,
   placeholder="Select Option",
)
if option == "Music Helper Chatbot":
  st.header(title_bot)
  if "messages" not in st.session_state.keys():
      st.session_state.messages = [{"role":"user","content":"Instructions: You are a chatbot for a music application. Respond with 'I can't answer that' to any question not related to music or music theory."},{"role":"assistant","content":"How can I help you?"}]

  for message in st.session_state.messages[2:]:
      with st.chat_message(message["role"]):
          st.write(message["content"])

  if prompt := st.chat_input():
      st.session_state.messages.append({"role": "user", "content": prompt})
      with st.chat_message("user"):
          st.write(prompt)


  if st.session_state.messages[-1]["role"] != "assistant":
      with st.chat_message("assistant"):
          with st.spinner("Thinking..."):
              response = assistant(prompt,st.session_state.messages)
              st.write(response)
      message = {"role": "assistant", "content": response}
      st.session_state.messages.append(message)
elif option == "Audio Generator":
  st.header(title_musicgen)
  prompt_musicgen = st.text_area("Describe your audio")
  duration_musicgen = st.number_input("Insert duration (seconds)", value=None, placeholder="Type a number...")
  generating = st.button('Start Generating!')
  if generating:
    if prompt_musicgen == "" or duration_musicgen == None or duration_musicgen == 0 or prompt_musicgen == None:
          generating = False
          st.error("Please provide a valid prompt and duration!")
    else:
        with st.spinner('Generating...'):
            sample = text_to_audio(prompt_musicgen,duration_musicgen)
            sample_rate = 32000
            st.success('Done!')
            st.audio(sample,sample_rate=sample_rate)
elif option == "Info":
    st.header("SynthScribe")
    st.subheader("AI Music chatbot and audio generation")
    st.text("Created by Vardan Petrodsyan, Ruben Khrimyan, Taron Seynyan, Aram Baghdasaryan")