from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from pydantic import BaseModel, EmailStr
from typing import List
import os
from dotenv import load_dotenv

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

    topic = inputText

    message = MessageSchema(
        subject="Update on " + topic,
        recipients=email_list,
        body=f"Dear colleagues, please find the latest updates on: {topic}",
        subtype="html"
    )

    fm = FastMail(conf)
    try:
        await fm.send_message(message)
        return templates.TemplateResponse("result.html", {"request": request, "message": f"Email sent successfully to {', '.join(email_list)}."})
    except Exception as e:
        return templates.TemplateResponse("result.html", {"request": request, "message": f"Failed to send email: {str(e)}."})

@app.get("/return/")
async def return_to_index(request: Request):
    return RedirectResponse(url='/')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
