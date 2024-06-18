# pip install requests
import requests

class WhisperAPI:
    
    __TOKEN = "fDH3LTifu53Wn5Uwlhzl9cGSnYlUjUoe"        
    __url = "https://api.lemonfox.ai/v1/audio/transcriptions"
    __headers = { "Authorization": __TOKEN }

    @staticmethod
    def getTranscription(file: str, translate: bool = True):
        data = {
            "response_format": "json",
            "translate": True
        }
        files = None
        if not file.startswith("http"):
            files = {"file": open(file, "rb")}
        
        if not files:    
            data["file"] = file
        response = requests.post(WhisperAPI.__url, headers=WhisperAPI.__headers, data=data, files=files)
        return response.json()

# Example
# print(WhisperAPI.getTranscription("./ar_test.mp3"))
