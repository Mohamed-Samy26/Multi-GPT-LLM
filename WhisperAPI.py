# pip install requests
import requests

class WhisperAPI:
    
    __TOKEN = "fDH3LTifu53Wn5Uwlhzl9cGSnYlUjUoe"        
    __url = "https://api.lemonfox.ai/v1/audio/transcriptions"
    __headers = { "Authorization": __TOKEN }


    # Doc
    
    @staticmethod
    def getTranscription(file: str, translate: bool = True)-> str:
        """
            file: str - Path to the audio file or URL
            translate: bool - Translate the transcription to English, Default: True
            return: str - Transcription text of the audio file
        """
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
        return response.json()['text']

# Example

# from WhisperAPI import WhisperAPI

# print(WhisperAPI.getTranscription("./ar_test.mp3")) # English translation
# print(WhisperAPI.getTranscription("./ar_test.mp3", False)) # No translation

