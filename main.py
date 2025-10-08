import os
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file

app = FastAPI()

# Initialize the Gradio Client
client = Client("ResembleAI/Chatterbox")

@app.post("/generate_tts_audio")
async def generate_tts_audio(
    text_input: str = Form(...),
    audio_prompt_path_input: UploadFile = File(...),
    exaggeration_input: float = Form(0.5),
    temperature_input: float = Form(0.8),
    seed_num_input: int = Form(0),
    cfgw_input: float = Form(0.5)
):
    try:
        # Ensure the temp directory exists
        temp_dir = "./temp/"
        os.makedirs(temp_dir, exist_ok=True)

        # Save the uploaded audio file locally
        file_path = os.path.join(temp_dir, audio_prompt_path_input.filename)
        with open(file_path, "wb") as file:
            content = await audio_prompt_path_input.read()
            file.write(content)

        # Handle the audio file with Gradio client
        audio_file = handle_file(file_path)

        # Call the Gradio endpoint
        result = client.predict(
            text_input=text_input,
            audio_prompt_path_input=audio_file,
            exaggeration_input=exaggeration_input,
            temperature_input=temperature_input,
            seed_num_input=seed_num_input,
            cfgw_input=cfgw_input,
            api_name="/generate_tts_audio"
        )

        # Clean up the local file
        os.remove(file_path)

        # Return the result
        return JSONResponse(content={"result": result})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
