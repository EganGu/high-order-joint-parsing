from __future__ import annotations
from typing import Dict, Iterable, List, Union, Tuple, Optional

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Curve(object):

    def __init__(
        self, 
        palette: str = 'muted', 
        dpi: int = 200, 
    ) -> Curve:
        self.palette = palette
        self.dpi = dpi
    
    def plot(
        self, 
        data: Union[pd.DataFrame, dict], 
        xy: Tuple(str, str) = ('', ''), 
        title: str = '', 
        path: Optional[str] = None, 
        yticks: Optional[Iterable[float]] = None, 
        show_value: bool = False, 
        value_interval: Optional[int] = None
    ) -> Curve:
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        data.columns = data.columns.map(lambda x:x.upper())
        palette = sns.color_palette(self.palette, len(data.columns))

        ax = sns.lineplot(data=data, palette=palette)
        self.set_desc(ax, xy, title, yticks)
        if show_value:
            self.set_value_text(data, value_interval, palette)
        
        if path is not None:
            plt.savefig(path, dpi=200)
        plt.cla()
        
    def set_desc(
        self, 
        ax: plt.Axes, 
        xy: Tuple(str, str), 
        title: str, 
        yticks: Optional[Iterable[float]]
    ):
        ax.set_xlabel(xy[0])
        ax.set_ylabel(xy[1])
        ax.set_title(title)
        if yticks is not None:
            ax.set_yticks(yticks)
    
    def set_value_text(
        self, 
        data: pd.DataFrame, 
        interval: Optional[int], 
        palette: sns.palettes._ColorPalette, 
        offset_type: str = 'half', 
    ):
        interval = interval if interval is not None else len(data) // 5
        offset = np.ones((len(data.columns))) * 0.01
        if offset_type == 'half':
            offset[:len(offset)//2] *= -1
        else:
            offset[::2] *= -1
        for i, col in enumerate(data.columns):
            y = list(data[col])
            for x, y in enumerate(y):
                if x % interval == 0:
                    plt.text(x=x, y=y+offset[i], s='{:.3f}'.format(y), color=palette[i])


class MetricCurve(Curve): 
    def __init__(self, palette: str = 'muted', dpi: int = 200) -> MetricCurve:
        super().__init__(palette, dpi)
    
    def plot(
        self, 
        data: Union[pd.DataFrame, dict], 
        xy: Tuple(str, str) = ('Epoch', 'Metric'), 
        title: str = '', 
        path: Optional[str] = None, 
        yticks: Optional[Iterable[float]] = [0.8+((0.02)*n) for n in range(11)], 
        show_value: bool = True, 
        value_interval: Optional[int] = None
    ) -> MetricCurve:
        return super().plot(data, xy, title, path, yticks, show_value, value_interval)
    
    def generate_cmp_data(
        self, 
        datas: Iterable[pd.DataFrame], 
        names: Iterable[str], 
        cmp_cols: List[str]
    ) -> MetricCurve:
        cmp_datas = [None] * len(datas)
        for i, data in enumerate(datas):
            cmp_datas[i] = data[cmp_cols]
            cmp_datas[i].columns = cmp_datas[i].columns.map(lambda x:x+' [%s]'%names[i])
        return pd.concat(cmp_datas, axis=1)

class LossCurve(Curve): 
    def __init__(self, palette: str = 'muted', dpi: int = 200) -> Curve:
        super().__init__(palette, dpi)
    
    def plot(
        self, 
        data: Union[pd.DataFrame, dict], 
        xy: Tuple(str, str) = ('Epoch', 'Loss'), 
        title: str = '', 
        path: Optional[str] = None, 
        yticks: Optional[Iterable[float]] = None, 
        show_value: bool = True, 
        value_interval: Optional[int] = None
    ) -> Curve:
        return super().plot(data, xy, title, path, yticks, show_value, value_interval)

