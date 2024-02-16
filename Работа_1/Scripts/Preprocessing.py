from Scripts.IPreprocess import IPreprocess


class RegressionDiamondPreprocessing(IPreprocess):
    def all_preprocess(self, iq_down=0.2, iq_up=0.8, flood="unknown") -> bool:
        self._headlines_preprocess()
        self._filtered_iqr(iq_down=iq_down, iq_up=iq_up)
        self._mean_flooding()
        self._flooding(flood=flood)
        return True


class ClassificationTransactionPreprocessing(IPreprocess):
    def all_preprocess(self, iq_down=0.2, iq_up=0.8, flood="unknown", int_type=None) -> bool:
        self._headlines_preprocess()
        self._mean_flooding()
        self._flooding(flood=flood)
        self._format_type(int_type, int)
        return True
