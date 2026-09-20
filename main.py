import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from google import genai
from google.genai.errors import APIError
from PIL import Image

app = FastAPI(
    title="API OCR Gemini",
    description="API FastAPI pour l'extraction de texte depuis des images à l'aide de Google Gemini.",
    version="1.1.0",
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


@app.get("/")
def read_root():
    """Vérification de l'état de l'API."""
    return {
        "status": "online",
        "message": "Bienvenue sur l'API OCR Gemini !",
        "api_key_configured": GEMINI_API_KEY is not None,
    }


@app.get("/models")
def list_available_models():
    """Affiche la liste dynamique des modèles supportés par votre clé API."""
    if not client:
        raise HTTPException(
            status_code=500,
            detail="La clé GEMINI_API_KEY n'est pas configurée.",
        )
    try:
        models = [m.name for m in client.models.list()]
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr")
async def extract_text_from_image(file: UploadFile = File(...)):
    """Reçoit une photo (JPG, PNG, WebP) et extrait tout le texte visible."""
    if not client:
        raise HTTPException(
            status_code=500,
            detail="La clé GEMINI_API_KEY n'est pas configurée dans les variables d'environnement.",
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Le fichier fourni doit être une image (ex: image/jpeg, image/png).",
        )

    try:
        # Ouverture de l'image
        image = Image.open(file.file)

        # Modèles valides (gemini-3.6-flash en principal)
        candidate_models = [
            "gemini-3.6-flash",
            "gemini-2.5-flash-lite",
        ]

        response = None
        last_error = None

        prompt = (
            "Extrais tout le texte visible dans cette image. "
            "Restitue uniquement le texte extrait sans aucun commentaire ni explication."
        )

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
                # Récupération sécurisée du code d'erreur HTTP
                err_code = getattr(err, "code", None)
                err_str = str(err)

                # Si le modèle est obsolète/introuvable (404) ou surchargé (503), on passe au suivant
                if err_code in [404, 503] or "404" in err_str or "503" in err_str:
                    continue
                # Si c'est une autre erreur API (ex: clé invalide 401), on stoppe immédiatement
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
