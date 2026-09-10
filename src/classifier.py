import os
from pydantic import BaseModel, Field
from google import genai
from PIL import Image
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

class WasteAnalysisResult(BaseModel):
    item_name: str = Field(description="Name of detected waste item")
    category: str = Field(description="One of: Wet, Recyclable, E-waste, Hazardous, General")
    material: str = Field(description="Material type, e.g., Plastic #1 PET, Organic Food, Lithium Battery")
    confidence: float = Field(description="Confidence from 0.0 to 1.0")
    condition: str = Field(description="e.g., Clean, Greasy/Contaminated, Damaged")
    disposal_bin: str = Field(description="Recommended bin color and name")
    preparation_steps: list[str] = Field(description="Step by step instructions before binning")
    upcycle_idea: str = Field(description="A creative DIY reuse idea")
    environmental_fact: str = Field(description="Environmental benefit or impact of properly disposing this")

def classify_waste(image: Image.Image) -> WasteAnalysisResult:
    # 1. Check local .env first, then fallback to Streamlit Cloud secrets
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            if "GEMINI_API_KEY" in st.secrets:
                api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env or Streamlit Secrets.")

    client = genai.Client(api_key=api_key)
    prompt = """
    You are an expert Environmental & Waste Management AI.
    Analyze the uploaded image of waste:
    1. Identify the primary object and material.
    2. Classify into: 'Wet', 'Recyclable', 'E-waste', 'Hazardous', or 'General'.
    3. Note if food contamination degrades recyclability (e.g., pizza grease).
    4. Provide clear disposal prep steps, creative upcycling, and an eco fact.
    """

    # 2. Uses the updated gemini-3.6-flash model
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[image, prompt],
        config={
            "response_mime_type": "application/json",
            "response_schema": WasteAnalysisResult,
        }
    )
    return response.parsed