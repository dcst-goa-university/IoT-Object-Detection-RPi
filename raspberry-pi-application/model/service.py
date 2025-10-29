from huggingface_hub import hf_hub_download
import numpy
import onnxruntime

class ModelService:
    def __init__(self, repo_id:str, file_name:str, revision:str):
        model_path = self._download_model(repo_id, file_name, revision)
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        
        
    def _download_model(self, repo_id:str, file_name:str, revision:str) -> str:
        model_path = hf_hub_download(repo_id=repo_id, filename=file_name, revision=revision)
        return model_path

    def infer(self, frame:numpy.ndarray) -> numpy.ndarray:
        preds = self.session.run(None, {'images': frame.astype(numpy.float32)})
        return preds
        
