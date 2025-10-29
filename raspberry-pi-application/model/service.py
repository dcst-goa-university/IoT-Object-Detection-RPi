from huggingface_hub import hf_hub_download
import numpy
import onnxruntime

class ModelService:
    def __init__(self, repo_id:str, file_name:str):
        self.repo_id = repo_id
        self.file_name = file_name
        self.model_path = self._download_model()
        self.session = onnxruntime.InferenceSession(
            self.model_path,
            providers=['CPUExecutionProvider']
        )
        
        
    def _download_model(self) -> str:
        model_path = hf_hub_download(repo_id=self.repo_id, filename=self.file_name)
        return model_path

    def infer(self, frame:numpy.ndarray) -> numpy.ndarray:
        preds = self.session.run(None, {'images': frame.astype(numpy.float32)})
        return preds
        
