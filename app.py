from PyQt5.QtWidgets import QMainWindow

from widgets        import CheckableComboBox      # noqa: F401  (used in ui_init_mixin)
from ui_init_mixin  import UIInitMixin
from data_mixin     import DataMixin
from stats_mixin    import StatsMixin
from ml_mixin       import MLMixin
from ensemble_mixin import EnsembleMixin
from viz_mixin      import VizMixin

class ExcelViewerQt(QMainWindow,
                    UIInitMixin,
                    DataMixin,
                    StatsMixin,
                    MLMixin,
                    EnsembleMixin,
                    VizMixin):
    def __init__(self):
        super().__init__()
        self.df                      = None
        self.filtered_df             = None
        self.last_predictions        = None
        self.last_actual             = None
        self.last_train_predictions  = None   # NEW — for train/test viz
        self.last_train_actual       = None   # NEW
        self.trained_model           = None
        self.trained_scaler          = None
        self.param_widgets           = {}
        self.ensemble_models         = []
        self.analysis_context        = {}
        self.initUI()