import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from google import genai
from google.genai.errors import APIError
from PIL import Image

app = FastAPI(
    title="API OCR avec Gemini",
    description="API permettant de convertir le texte d'une photo à l'aide de l'API Google Gemini 3.6 Flash.",
    version="1.0.0",
)

# Récupération de la clé API depuis les variables d'environnement
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialisation du client Google GenAI si la clé est disponible
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None


@app.get("/")
def read_root():
    """Route de vérification de l'état du service (Health check)."""
    return {
        "status": "online",
        "message": "Bienvenue sur l'API OCR Gemini !",
        "api_key_configured": GEMINI_API_KEY is not None,
    }


@app.post("/ocr")
async def extract_text_from_image(file: UploadFile = File(...)):
    if not client:
        raise HTTPException(
            status_code=500,
            detail="La clé GEMINI_API_KEY n'est pas configurée.",
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Le fichier fourni doit être une image.",
        )

    try:
        image = Image.open(file.file)

        # Liste des modèles à essayer par ordre de préférence
        candidate_models = [
            "gemini-3.6-flash",
            "gemini-1.5-flash",
            "gemini-2.5-flash-lite",
        ]

        response = None
        last_error = None

        prompt = (
            "Extrais tout le texte visible dans cette image. "
            "Restitue uniquement le texte extrait sans commentaires, ni explications."
        )

        # Essayer chaque modèle jusqu'à ce qu'un fonctionne
        for model_name in candidate_models:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt, image],
                )
                # Si l'appel réussit, on sort de la boucle
                break
            except APIError as err:
                last_error = err
                # Si erreur 503, on passe au modèle suivant
                if getattr(err, "code", None) == 503 or "503" in str(err):
                    continue
                raise err

        if not response:
            raise HTTPException(
                status_code=503,
                detail=f"Tous les modèles sont actuellement surchargés : {str(last_error)}",
            )

        return {
            "filename": file.filename,
            "extracted_text": response.text.strip() if response.text else "",
        }

    except APIError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Erreur de communication avec l'API Gemini : {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement de l'image : {str(e)}",
        )
