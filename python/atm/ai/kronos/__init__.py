import logging

from .kronos import Kronos, KronosPredictor, KronosTokenizer

logger = logging.getLogger(__name__)

model_dict = {
    'kronos_tokenizer': KronosTokenizer,
    'kronos': Kronos,
    'kronos_predictor': KronosPredictor
}


def get_model_class(model_name: str):
    """
    Get model class by name.

    Args:
        model_name: Name of the model to retrieve.

    Returns:
        Model class.

    Raises:
        NotImplementedError: If model_name is not found in model_dict.
    """
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        logger.error(f"Model {model_name} not found in model_dict")
        raise NotImplementedError(f"Model {model_name} not found. Available models: {list(model_dict.keys())}")


