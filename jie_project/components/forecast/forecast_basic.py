from ..models import SMETimesLlama, TimesLlama, TimesMIMO, TimesFusionLlama, TimesLlamaAblation


class ForecastBasic:
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'SMETimes_Llama': SMETimesLlama,
            'Times_Llama': TimesLlama,
            'Times_MIMO': TimesMIMO,
            "TimesFusion_Llama": TimesFusionLlama,
            'Times_Llama_Ablation': TimesLlamaAblation,
        }
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self, flag):
        pass

    def vali(self, vali_data, vali_loader, criterion, is_test=False):
        pass

    def train(self, setting):
        pass

    def test(self, setting, test=0):
        pass
