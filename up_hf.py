import os
from huggingface_hub import HfApi
api = HfApi()
repo_id = "zbller/Mecari"
api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", private=False, exist_ok=True, token=os.environ["HF_TOKEN"])
api.upload_folder(
    repo_id=repo_id,
    repo_type="space",
    folder_path=".",
    path_in_repo=".",
    ignore_patterns=[
        ".git", ".git/**", ".venv", ".venv/**", "__pycache__", "**/__pycache__",
        "KWDLC", "KWDLC/**", "annotations", "annotations/**", "experiments", "experiments/**",
        "mecari_morpheme.egg-info", "mecari_morpheme.egg-info/**",
    ],
    token=os.environ["HF_TOKEN"],
)
print(f"Uploaded to https://huggingface.co/spaces/{repo_id}")