import base64
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class GoogleClient:
    def __init__(self, model="gemma-3-1b-it", api_key=None, region="global"):
        client_options = {}
        if region == "us-central1":
            client_options["api_endpoint"] = "us-central1-generativelanguage.googleapis.com"
        
        self.client = genai.Client(
            api_key=api_key or os.environ.get("GEMINI_API_KEY"),
            **client_options
        )
        self.model = model
    
    def list_models(self):
        """List available models"""
        models = self.client.models.list()
        model_list = {model.model_dump()['name'] for model in models}
        return model_list
    
    def generate_content(self, text, response_schema=None):
        """Generate content (non-streaming)"""
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=text),
                ],
            ),
        ]
        
        config = {"response_mime_type": "text/plain"}
        if response_schema:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = response_schema
        
        generate_content_config = types.GenerateContentConfig(**config)
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=generate_content_config,
        )
        
        return response.text
    
    def generate_content_stream(self, text, response_schema=None):
        """Generate content (streaming)"""
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=text),
                ],
            ),
        ]
        
        config = {"response_mime_type": "text/plain"}
        if response_schema:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = response_schema
        
        generate_content_config = types.GenerateContentConfig(**config)
        
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=generate_content_config,
        ):
            yield chunk.text