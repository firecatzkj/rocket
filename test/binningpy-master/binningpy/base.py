import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    check_is_fitted
)


class BinningBase(BaseEstimator, TransformerMixin):
    """分箱的基类.

    实现transform和inverse_transform两个方法和__init__方法,需要对象有_bins对象

    Public:
        bin_nbr (int): 分箱的个数
        confined (bool): 是否以训练数据的上下限位上下限
        copy (bool): 是否复制输入

    Protected:
        _bins (List[float]): 分箱的间隔位置组成的list,相邻两点左开右闭

    """

    def __init__(self, bin_nbr: int=4, confined: bool=True, copy: bool=True)->None:
        if not isinstance(bin_nbr, int):
            raise AttributeError("bin number must be int")
        if confined and bin_nbr <= 0:
            raise AttributeError("bin number must > 0 when confined is True")
        if not confined and bin_nbr <= 2:
            raise AttributeError("bin number must > 2 when confined is False")

        self.bin_nbr = bin_nbr
        self.confined = confined
        self.copy = copy

    def _transform_item(self, item, features_line)->int:
        bins = self._bins[features_line]
        for i in range(len(bins) - 1):
            if self.confined:
                if bins[i] <= item < bins[i + 1]:
                    return i
                else:
                    continue
            else:
                if bins[i] <= item < bins[i + 1]:
                    return i + 1
                else:
                    continue
        else:
            if self.confined:
                raise AttributeError(f"{item} not in range")
            else:
                if item >= bins[-1]:
                    return len(bins)
                if item < bins[0]:
                    return 0

    def _transform(self, x, features_line):
        for i, value in enumerate(x):
            x[i] = self._transform_item(value, features_line)
        return x

    def transform(self, X):
        """连续数据变换为离散值.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Input data that will be transformed.
        """
        check_is_fitted(self, '_bins')

        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES)
        result = []
        for features_line, x in enumerate(X.T):
            result.append(self._transform(x, features_line))
        return np.array(result, dtype=int).T

    def _inverse_transform_item(self, item, features_line):
        bins = self._bins[features_line]
        #print(item)
        if self.confined:
            return (bins[item],bins[item+1])
        else:
            if item == 0:
                return (-np.inf,bins[item])
            elif item == len(bins):
                return (bins[-1],np.inf)
            else:
                return (bins[item-1],bins[item])
        

    def _inverse_transform(self,x, features_line):
        result = []
        for i, value in enumerate(x):
            result.append(self._inverse_transform_item(value, features_line))
        return result

    def inverse_transform(self, X):
        """逆变换.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Input data that will be transformed. It cannot be sparse.
        """
        check_is_fitted(self, '_bins')

        X = check_array(X, copy=self.copy)
        result = []
        for features_line, x in enumerate(X.T):
            result.append(self._inverse_transform(x, features_line))
        return result

    def fit(self, X, y=None)->None:
        """训练,未实现.

        Args:
            X (Sequence[float]): 待训练的连续型参数
            y (Optional[Sequence[float]]): Defaults to None. 没用

        """
        raise NotImplemented


__all__ = ["BinningBase"]
