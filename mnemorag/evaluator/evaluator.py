import os

from mnemorag.evaluator.metrics import ExactMatch, F1_Score, Sub_ExactMatch


class Evaluator:
    """Evaluate predictions with the configured metrics."""

    def __init__(self, config):
        self.config = config
        self.save_dir = config["save_dir"]

        self.save_metric_flag = config["save_metric_score"]
        self.save_data_flag = config["save_intermediate_data"]
        self.metrics = [metric.lower() for metric in self.config["metrics"]]

        available = {"em": ExactMatch, "f1": F1_Score, "acc": Sub_ExactMatch}
        unknown = set(self.metrics) - available.keys()
        if unknown:
            raise ValueError(f"Unsupported metrics: {sorted(unknown)}")
        self.metric_class = {name: available[name](config) for name in self.metrics}

    def evaluate(self, data):
        """Compute per-request scores and return aggregate metrics."""

        result_dict = {}
        for metric in self.metrics:
            metric_result, metric_scores = self.metric_class[metric].calculate_metric(
                data
            )
            if len(metric_scores) != len(data):
                raise ValueError(f"{metric}: expected one score per request")
            result_dict.update(metric_result)
            for metric_score, item in zip(metric_scores, data):
                item.update_evaluation_score(metric, metric_score)

        if self.save_metric_flag:
            self.save_metric_score(result_dict)

        if self.save_data_flag:
            self.save_data(data)

        return result_dict

    def save_metric_score(self, result_dict, file_name="metric_score.txt"):
        save_path = os.path.join(self.save_dir, file_name)
        with open(save_path, "w", encoding="utf-8") as f:
            for k, v in result_dict.items():
                f.write(f"{k}: {v}\n")

    def save_data(self, data, file_name="intermediate_data.json"):
        """Save evaluated requests and their scores."""

        save_path = os.path.join(self.save_dir, file_name)

        data.save(save_path)
