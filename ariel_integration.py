from ariel_algorithm import ARIELAlgorithm
from ariel_training import train_model
from ariel_logging import ARIELLogger

class ARIELInterface:
    def __init__(self):
        self.algorithm = ARIELAlgorithm()
        self.logger = ARIELLogger()

    def create_and_train_llm(self, config):
        model = self.algorithm.create_model(config)
        trained_model = train_model(model, config)
        self.logger.log_training_results(trained_model)
        return trained_model

    def evaluate_llm(self, model):
        return self.algorithm.evaluate(model)