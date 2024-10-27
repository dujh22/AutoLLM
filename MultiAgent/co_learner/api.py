from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from user_manager import UserManager
from textToVoiceAgent import text2video

app = FastAPI()

# Assuming you have a UserManagement class with an add_user method.
user_management = UserManager()

class UserCreateRequest(BaseModel):
    phone_number: str = Field(..., example="1234567890")
    news_link: Optional[list] = Field(None, example="http://example.com")
    zzh_link: Optional[list] = Field(None, example="http://example.com")
    paper_link: Optional[list] = Field(None, example="http://example.com")
    prompt: Optional[str] = Field(None, example="Preferred prompt")

@app.post("/add_user")
def add_user(request: UserCreateRequest):
    try:
        if user_management.user_exists(request.phone_number) == False:
            user_management.create_user(request.phone_number)
        if request.news_link:
            user_management.reset_link(request.phone_number, request.news_link, "news")
        if request.zzh_link:
            user_management.reset_link(request.phone_number, request.zzh_link, "gzh")
        if request.paper_link:
            user_management.reset_link(request.phone_number, request.paper_link, "paper")
        if request.prompt:
            user_management.reset_preference(request.phone_number, request.prompt)
        
        return {"status": "success", "message": "User added successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
class UserChatRequest(BaseModel):
    phone_number: str = Field(..., example="1234567890")
    message: str = Field(..., example="Hello, how are you?")

@app.post("/chat")
def chat(request: UserChatRequest):
    try:
        # Here we call the chat method from your UserManagement class
        responseWithLink, responseWithoutLink = user_management.interact(request.phone_number, request.message)
        video = text2video(responseWithoutLink)
        video_path = video["video_path"]
        link_path = video["link_path"]
        return {"status": "success", "message": responseWithLink, "video_path": video_path, "link_path": link_path}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/chat2")
def chat2(request: UserChatRequest):
    try:
        # Here we call the chat method from your UserManagement class
        responseWithLink, responseWithoutLink = user_management.interact(request.phone_number, request.message)
        return {"status": "success", "message": responseWithLink}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/get_user")
def get_user(phone_number: str):
    try:
        # Here we call the get_user method from your UserManagement class
        if user_management.user_exists(phone_number) == True:
            return {
                "status": "success",
                "phone_number": phone_number,
                "news_link": user_management.get_links(phone_number, "news"),
                "zzh_link": user_management.get_links(phone_number, "gzh"),
                "paper_link": user_management.get_links(phone_number, "paper"),
                "prompt": user_management.get_preference(phone_number)
            }
        else:
            return {"status": "error", "message": "User not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
@app.get("/get_briefings")
def get_briefings(phone_number: str):
    try:
        # Here we call the get_user method from your UserManagement class
        if user_management.user_exists(phone_number) == True:
            return {
                "status": "success",
                "phone_number": phone_number,
                "briefings": user_management.get_briefing(phone_number)
            }
        else:
            return {"status": "error", "message": "User not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# To run the app, use:
# uvicorn filename:app --reload
# uvicorn api:app --host 0.0.0.0 --port 8000 --reload