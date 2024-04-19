from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from pydantic import BaseModel, EmailStr
from typing import List
import os
from dotenv import load_dotenv

import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel


tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", use_auth_token='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')
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", load_in_4bit=True, torch_dtype=torch.float16, device_map="auto", use_auth_token='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')
tokenizer.pad_token = "!"

sys_msg = "Given the description of an email task, identify the intended recipients and generate a relevant topic for the email based on the given details. Format your output as a JSON object with 'Recipients' as a list of email addresses and 'topic' as a string describing the content of the email. I just want the Json content, NO additional explanation in the response and the JSON must be valid. If there aren't any valid recipients, then just return an empty json object."


load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class EmailSchema(BaseModel):
    Recipients: List[EmailStr]
    topic: str

conf = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_FROM=os.getenv("MAIL_FROM"),
    MAIL_PORT=int(os.getenv("MAIL_PORT")),
    MAIL_SERVER=os.getenv("MAIL_SERVER"),
    MAIL_FROM_NAME=os.getenv("MAIL_FROM_NAME"),
    MAIL_STARTTLS = True,
    MAIL_SSL_TLS = False,
    USE_CREDENTIALS = True,
    VALIDATE_CERTS = True
)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/send-email/")
async def send_email(request: Request, inputText: str = Form(...)):
    email_list = ['daniel.hall@live.se']
    # email_list = recipients.split(",")
    # if not email_list:
    #     return templates.TemplateResponse("result.html", {"request": request, "message": "No recipients provided."})

    p = "<s> [INST]" + sys_msg +"\n"+ inputText + "[/INST] </s>"

    with torch.no_grad():
        input_ids = tokenizer([p], return_tensors="pt")
        generated_ids = model.generate(**input_ids,max_new_tokens=100, do_sample=True)
        tokenizer.batch_decode(generated_ids)[0]

    topic = tokenizer.batch_decode(generated_ids)[0]

    message = MessageSchema(
        subject="Here a generated snipppet!",
        recipients=email_list,
        body=f"Generated output {topic}",
        subtype="html"
    )

    fm = FastMail(conf)
    try:
        await fm.send_message(message)
        return templates.TemplateResponse("result.html", {"request": request, "message": f"Email with message {message} sent successfully to {', '.join(email_list)}."})
    except Exception as e:
        return templates.TemplateResponse("result.html", {"request": request, "message": f"Failed to send email: {str(e)}."})

@app.get("/return/")
async def return_to_index(request: Request):
    return RedirectResponse(url='/')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
