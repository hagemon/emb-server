import yaml
from threading import Lock

class Config:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._load_config()
            return cls._instance

    def _load_config(self):
        with open('config.yaml', 'r') as file:
            self._config = yaml.safe_load(file)

    def get(self, key, default=None):
        return self._config.get(key, default)

# 全局访问点
config = Config()

if __name__ == "__main__":
    print(config.get("base_path"))
    print(config.get("model_path"))
