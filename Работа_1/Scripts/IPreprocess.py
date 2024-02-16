from abc import ABCMeta, abstractmethod


class IPreprocess(metaclass=ABCMeta):
    def __init__(self, data):
        self.data = data

    def __call__(self):
        return self.data

    @abstractmethod
    def all_preprocess(self, iq_down=0.2, iq_up=0.8, flood="unknown") -> bool:
        """Набор всех необходимых функций для препроцесса"""
        return True

    def _headlines_preprocess(self) -> bool:
        """Приведение заголовков к нормальному виду"""
        self.data.columns = list("_".join(x.strip().lower().split()) for x in self.data.columns)
        return True

    def _filtered_iqr(self, iq_down=0.2, iq_up=0.8) -> bool:
        """Очистка данных по процентилям"""
        q1 = self.data.quantile(iq_down)
        q2 = self.data.quantile(iq_up)
        iqr = q2 - q1
        self.data = self.data[~((self.data < (q1 - 1.5 * iqr)) | (self.data > (q2 + 1.5 * iqr))).any(axis=1)]
        return True

    def _flooding(self, flood="unknown") -> bool:
        """Заполнение всех пропущенных значений ключевым словом"""
        for i in self.data.columns:
            self.data[i] = self.data[i].fillna(flood)
        return True

    def _mean_flooding(self) -> bool:
        """Заполнение всех пропущенных данных в числовых столбцах средним значением"""
        for i in self.data.columns:
            if isinstance(self.data[i].dtype, (float, int)):
                s = self.data[i].mean()
                self.data[i] = self.data[i].fillna(s)
        return True

    def _format_type(self, lst_column=None, datatype=object) -> bool:
        """Приведенеие столбцов к заданному типу"""
        if isinstance(lst_column, (list, tuple)):
            for column in lst_column:
                self.data[column] = self.data[column].astype(datatype)
        return True

