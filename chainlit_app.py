import chainlit as cl
from main import run_summary
from grok_client import GrokLLM
from vfm_chains import run_caption

@cl.on_message
async def handle_msg(msg):
    content = msg.content
    if content.startswith("summarize:"):
        summary = run_summary(content.replace("summarize:", ""))
        await cl.Message(content=summary).send()
    elif content.startswith("grok:"):
        grok = GrokLLM()
        reply = grok(content.replace("grok:", ""))
        await cl.Message(content=reply).send()

@cl.on_file_upload
async def handle_upload(file):
    caption = run_caption(file.path)
    await cl.Message(content=f"Image Caption: {caption}").send()