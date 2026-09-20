import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from google import genai
from google.genai.errors import APIError
from PIL import Image

app = FastAPI(title="API OCR Gemini")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


@app.post("/ocr")
async def extract_text_from_image(file: UploadFile = File(...)):
    if not client:
        raise HTTPException(
            status_code=500,
            detail="La clé GEMINI_API_KEY n'est pas configurée.",
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Le fichier fourni doit être une image."
        )

    try:
        image = Image.open(file.file)

        # Liste des identifiants valides à essayer successivement
        candidate_models = ["gemini-2.5-flash", "gemini-2.0-flash"]

        response = None
        last_error = None

        prompt = "Extrais tout le texte visible dans cette image. Restitue uniquement le texte extrait sans aucun commentaire ni explication."

        for model_name in candidate_models:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt, image],
                )
                if response and response.text:
                    break
            except APIError as err:
                last_error = err
                # On retente avec le modèle suivant si 404 (non trouvé) ou 503 (surchargé)
                if err.code in [404, 503] or any(
                    code in str(err) for code in ["404", "503"]
                ):
                    continue
                raise err

        if not response or not response.text:
            raise HTTPException(
                status_code=502,
                detail=f"Impossible de traiter l'image avec les modèles disponibles : {str(last_error)}",
            )

        return {
            "filename": file.filename,
            "extracted_text": response.text.strip(),
        }

    except APIError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Erreur de communication avec l'API Gemini : {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur lors du traitement : {str(e)}"
        )
