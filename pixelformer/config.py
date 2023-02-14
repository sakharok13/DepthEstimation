from dataclasses import dataclass


@dataclass
class Config:
    mode: str = 'train'
    model_name: str = 'pixelformer'
    encoder: str = 'large07'
    pretrain: str = 'None'
    data_path: str = 'my_data_path'
    height: int = 480
    width: int = 640
    max_depth: int = 10 
    checkpoint_path: str = 'checkpoint_path'
    adam_eps: float = 1e-6
    batch_size: int = 4
    num_epochs: int = 50
    variance_focus: float = 0.85
    learning_rate: float = 1e-4
    gpu_count: int = 1
    project: str = 'DepthEstimation_experiment_'
    weight_decay: float = 1e-5