from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QComboBox, QLineEdit, QTextEdit, QTabWidget,
                             QGroupBox, QCheckBox, QScrollArea, QSpinBox,
                             QListWidget, QRadioButton, QButtonGroup, QTextBrowser)
from PyQt5.QtGui import QFont
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

def _check_transformers():
    try:
        import transformers  # noqa: F401
        import torch         # noqa: F401
        return True
    except ImportError:
        return False

class UIInitMixin:
    def initUI(self):
        self.setWindowTitle('Excel 智能分析器 - AI驱动 (优化版)')
        self.setGeometry(100, 100, 1400, 800)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        button_layout = QHBoxLayout()
        self.btn_open = QPushButton('📂 打开Excel文件')
        self.btn_open.clicked.connect(self.load_excel)
        self.btn_open.setMinimumHeight(35)
        button_layout.addWidget(self.btn_open)

        self.btn_export = QPushButton('💾 导出为CSV')
        self.btn_export.clicked.connect(self.export_csv)
        self.btn_export.setMinimumHeight(35)
        self.btn_export.setEnabled(False)
        button_layout.addWidget(self.btn_export)

        self.btn_refresh = QPushButton('🔄 刷新')
        self.btn_refresh.clicked.connect(self.refresh_data)
        self.btn_refresh.setMinimumHeight(35)
        self.btn_refresh.setEnabled(False)
        button_layout.addWidget(self.btn_refresh)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel('搜索列:'))
        self.combo_columns = QComboBox()
        self.combo_columns.setMinimumWidth(150)
        filter_layout.addWidget(self.combo_columns)
        filter_layout.addWidget(QLabel('关键词:'))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText('输入搜索关键词...')
        self.search_input.textChanged.connect(self.filter_data)
        self.search_input.setMinimumWidth(200)
        filter_layout.addWidget(self.search_input)
        self.btn_clear_filter = QPushButton('✖ 清除筛选')
        self.btn_clear_filter.clicked.connect(self.clear_filter)
        filter_layout.addWidget(self.btn_clear_filter)
        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        self.tabs = QTabWidget()
        self.data_tab = QWidget()
        data_layout = QVBoxLayout()
        from PyQt5.QtWidgets import QTableWidget
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.setStyleSheet("""
            QTableWidget { gridline-color: #d0d0d0; background-color: white; }
            QTableWidget::item { padding: 5px; }
            QHeaderView::section { background-color: #5a6c7d; color: white; padding: 8px;
                font-weight: bold; border: 1px solid #4a5c6d; }
            QHeaderView::section:hover { background-color: #6b7d8e; }
        """)
        data_layout.addWidget(self.table)
        self.data_tab.setLayout(data_layout)

        self.stats_tab = QWidget()
        self.init_stats_tab()
        self.ml_tab = QWidget()
        self.init_ml_tab()
        self.ensemble_tab = QWidget()
        self.init_ensemble_tab()
        self.viz_tab = QWidget()
        self.init_viz_tab()
        self.ai_tab = QWidget()
        self.init_ai_tab()

        self.tabs.addTab(self.data_tab,     "📊 Data Review")
        self.tabs.addTab(self.stats_tab,    "📈 Statistics Analysis")
        self.tabs.addTab(self.ml_tab,       "🤖 Machine Learning")
        self.tabs.addTab(self.ensemble_tab, "🎯 Ensemble Model")
        self.tabs.addTab(self.viz_tab,      "📉 Visualization")
        # self.tabs.addTab(self.ai_tab,       "🧠 LLM")
        layout.addWidget(self.tabs)
        central_widget.setLayout(layout)
        self.statusBar().showMessage('就绪 - 请打开Excel文件')
        self.setStyleSheet("""
            QPushButton { background-color: #5a6c7d; color: white; border: none;
                padding: 8px 15px; border-radius: 4px; font-weight: 500; }
            QPushButton:hover { background-color: #6b7d8e; }
            QPushButton:pressed { background-color: #4a5c6d; }
            QPushButton:disabled { background-color: #d0d0d0; color: #888888; }
            QGroupBox { font-weight: 600; border: 2px solid #5a6c7d; border-radius: 5px;
                margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { color: #5a6c7d; }
        """)

    def init_stats_tab(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    
        layout = QVBoxLayout()
    
        button_group = QHBoxLayout()
        self.btn_desc_stats = QPushButton('📊 描述性统计')
        self.btn_desc_stats.clicked.connect(self.show_descriptive_stats)
        button_group.addWidget(self.btn_desc_stats)
        self.btn_correlation = QPushButton('🔗 相关性分析')
        self.btn_correlation.clicked.connect(self.show_correlation)
        button_group.addWidget(self.btn_correlation)
        self.btn_normality = QPushButton('📐 正态分布检验')
        self.btn_normality.clicked.connect(self.test_normality)
        button_group.addWidget(self.btn_normality)
        self.btn_outliers = QPushButton('🎯 异常值检测')
        self.btn_outliers.clicked.connect(self.detect_outliers)
        button_group.addWidget(self.btn_outliers)
        self.btn_distribution = QPushButton('📊 字段分布观察')
        self.btn_distribution.clicked.connect(self.show_distribution_analysis)
        button_group.addWidget(self.btn_distribution)
        button_group.addStretch()
        layout.addLayout(button_group)
    
        # 文字结果区（描述统计 / 相关性 / 正态检验 / 异常值）
        self.stats_result = QTextEdit()
        self.stats_result.setReadOnly(True)
        self.stats_result.setFont(QFont("Courier New", 9))
        layout.addWidget(self.stats_result)
    
        # 图表区（字段分布观察专用），默认隐藏
        self.stats_figure = Figure(figsize=(14, 8), dpi=100)
        self.stats_canvas = FigureCanvas(self.stats_figure)
        self.stats_canvas.hide()
        layout.addWidget(self.stats_canvas)
    
        self.stats_tab.setLayout(layout)   # ← setLayout 必须在最后

    def init_ml_tab(self):
        main_layout = QHBoxLayout()
        control_panel = QVBoxLayout()

        task_group = QGroupBox("1️⃣ 选择任务类型")
        task_layout = QVBoxLayout()
        self.ml_task_combo = QComboBox()
        self.ml_task_combo.addItems(['分类 (Classification)', '回归 (Regression)', '聚类 (Clustering)'])
        self.ml_task_combo.currentIndexChanged.connect(self.update_ml_models)
        task_layout.addWidget(self.ml_task_combo)
        task_group.setLayout(task_layout)
        control_panel.addWidget(task_group)

        model_group = QGroupBox("2️⃣ 选择模型")
        model_layout = QVBoxLayout()
        self.ml_model_combo = QComboBox()
        self.ml_model_combo.currentIndexChanged.connect(self.update_hyperparameters)
        model_layout.addWidget(self.ml_model_combo)
        model_group.setLayout(model_layout)
        control_panel.addWidget(model_group)

        self.param_group = QGroupBox("⚙️ 超参数调整 (可选)")
        self.param_scroll = QScrollArea()
        self.param_scroll.setWidgetResizable(True)
        self.param_scroll.setMaximumHeight(200)
        self.param_widget = QWidget()
        self.param_layout = QVBoxLayout()
        self.param_widget.setLayout(self.param_layout)
        self.param_scroll.setWidget(self.param_widget)
        param_group_layout = QVBoxLayout()
        param_group_layout.addWidget(self.param_scroll)
        self.param_group.setLayout(param_group_layout)
        control_panel.addWidget(self.param_group)

        feature_group = QGroupBox("3️⃣ 选择特征列")
        feature_layout = QVBoxLayout()
        feature_btn_layout = QHBoxLayout()
        self.btn_select_all = QPushButton('✓ 全选')
        self.btn_select_all.clicked.connect(self.select_all_features)
        feature_btn_layout.addWidget(self.btn_select_all)
        self.btn_select_none = QPushButton('✗ 全不选')
        self.btn_select_none.clicked.connect(self.select_none_features)
        feature_btn_layout.addWidget(self.btn_select_none)
        feature_layout.addLayout(feature_btn_layout)
        feature_scroll = QScrollArea()
        feature_scroll.setWidgetResizable(True)
        feature_scroll.setMaximumHeight(120)
        self.feature_widget = QWidget()
        self.feature_checkboxes_layout = QVBoxLayout()
        self.feature_widget.setLayout(self.feature_checkboxes_layout)
        feature_scroll.setWidget(self.feature_widget)
        feature_layout.addWidget(feature_scroll)
        feature_group.setLayout(feature_layout)
        control_panel.addWidget(feature_group)

        target_group = QGroupBox("4️⃣ 选择目标列")
        target_layout = QVBoxLayout()
        self.ml_target_combo = QComboBox()
        target_layout.addWidget(self.ml_target_combo)
        target_group.setLayout(target_layout)
        control_panel.addWidget(target_group)
        self.target_group = target_group

        cluster_group = QGroupBox("4️⃣ 聚类参数")
        cluster_layout = QHBoxLayout()
        cluster_layout.addWidget(QLabel("聚类数量:"))
        self.cluster_n = QSpinBox()
        self.cluster_n.setMinimum(2)
        self.cluster_n.setMaximum(20)
        self.cluster_n.setValue(3)
        cluster_layout.addWidget(self.cluster_n)
        cluster_group.setLayout(cluster_layout)
        control_panel.addWidget(cluster_group)
        self.cluster_group = cluster_group
        self.cluster_group.hide()

        model_buttons_layout = QHBoxLayout()
        self.btn_train = QPushButton('🚀 训练模型')
        self.btn_train.clicked.connect(self.train_model)
        self.btn_train.setMinimumHeight(40)
        model_buttons_layout.addWidget(self.btn_train)
        self.btn_reset = QPushButton('🔄 重置结果')
        self.btn_reset.clicked.connect(self.reset_ml_results)
        self.btn_reset.setMinimumHeight(40)
        self.btn_reset.setEnabled(False)
        model_buttons_layout.addWidget(self.btn_reset)
        control_panel.addLayout(model_buttons_layout)

        model_buttons_layout2 = QHBoxLayout()
        self.btn_save_model = QPushButton('💾 导出模型')
        self.btn_save_model.clicked.connect(self.save_model)
        self.btn_save_model.setMinimumHeight(40)
        self.btn_save_model.setEnabled(False)
        model_buttons_layout2.addWidget(self.btn_save_model)
        self.btn_load_model = QPushButton('📂 导入模型')
        self.btn_load_model.clicked.connect(self.load_model)
        self.btn_load_model.setMinimumHeight(40)
        model_buttons_layout2.addWidget(self.btn_load_model)
        control_panel.addLayout(model_buttons_layout2)
        control_panel.addStretch()

        result_layout = QVBoxLayout()
        self.ml_result = QTextEdit()
        self.ml_result.setReadOnly(True)
        self.ml_result.setFont(QFont("Courier New", 9))
        result_layout.addWidget(self.ml_result)

        main_layout.addLayout(control_panel, 1)
        main_layout.addLayout(result_layout, 2)
        self.ml_tab.setLayout(main_layout)
        self.update_ml_models()

    def init_ensemble_tab(self):
        main_layout = QHBoxLayout()
        control_panel = QVBoxLayout()

        task_group = QGroupBox("1️⃣ 选择任务类型")
        task_layout = QVBoxLayout()
        self.ensemble_task_combo = QComboBox()
        self.ensemble_task_combo.addItems(['分类 (Classification)', '回归 (Regression)'])
        task_layout.addWidget(self.ensemble_task_combo)
        task_group.setLayout(task_layout)
        control_panel.addWidget(task_group)

        model_manage_group = QGroupBox("2️⃣ 模型管理")
        model_manage_layout = QVBoxLayout()
        self.btn_add_model = QPushButton('➕ 添加模型')
        self.btn_add_model.clicked.connect(self.add_ensemble_model)
        model_manage_layout.addWidget(self.btn_add_model)
        self.ensemble_models_list = QListWidget()
        self.ensemble_models_list.setMaximumHeight(150)
        model_manage_layout.addWidget(QLabel('已添加的模型:'))
        model_manage_layout.addWidget(self.ensemble_models_list)
        self.btn_remove_model = QPushButton('➖ 移除选中模型')
        self.btn_remove_model.clicked.connect(self.remove_ensemble_model)
        model_manage_layout.addWidget(self.btn_remove_model)
        model_manage_group.setLayout(model_manage_layout)
        control_panel.addWidget(model_manage_group)

        feature_group = QGroupBox("3️⃣ 选择特征列")
        feature_layout = QVBoxLayout()
        feature_btn_layout = QHBoxLayout()
        self.btn_ensemble_select_all = QPushButton('✓ 全选')
        self.btn_ensemble_select_all.clicked.connect(self.select_all_ensemble_features)
        feature_btn_layout.addWidget(self.btn_ensemble_select_all)
        self.btn_ensemble_select_none = QPushButton('✗ 全不选')
        self.btn_ensemble_select_none.clicked.connect(self.select_none_ensemble_features)
        feature_btn_layout.addWidget(self.btn_ensemble_select_none)
        feature_layout.addLayout(feature_btn_layout)
        feature_scroll = QScrollArea()
        feature_scroll.setWidgetResizable(True)
        feature_scroll.setMaximumHeight(120)
        self.ensemble_feature_widget = QWidget()
        self.ensemble_feature_checkboxes_layout = QVBoxLayout()
        self.ensemble_feature_widget.setLayout(self.ensemble_feature_checkboxes_layout)
        feature_scroll.setWidget(self.ensemble_feature_widget)
        feature_layout.addWidget(feature_scroll)
        feature_group.setLayout(feature_layout)
        control_panel.addWidget(feature_group)

        target_group = QGroupBox("4️⃣ 选择目标列")
        target_layout = QVBoxLayout()
        self.ensemble_target_combo = QComboBox()
        target_layout.addWidget(self.ensemble_target_combo)
        target_group.setLayout(target_layout)
        control_panel.addWidget(target_group)

        ensemble_buttons_layout = QHBoxLayout()
        self.btn_train_ensemble = QPushButton('🚀 训练集成模型')
        self.btn_train_ensemble.clicked.connect(self.train_ensemble_model)
        self.btn_train_ensemble.setMinimumHeight(40)
        ensemble_buttons_layout.addWidget(self.btn_train_ensemble)
        self.btn_reset_ensemble = QPushButton('🔄 重置结果')
        self.btn_reset_ensemble.clicked.connect(self.reset_ensemble_results)
        self.btn_reset_ensemble.setMinimumHeight(40)
        self.btn_reset_ensemble.setEnabled(False)
        ensemble_buttons_layout.addWidget(self.btn_reset_ensemble)
        control_panel.addLayout(ensemble_buttons_layout)

        ensemble_io_layout = QHBoxLayout()
        self.btn_save_ensemble = QPushButton('💾 导出集成模型')
        self.btn_save_ensemble.clicked.connect(self.save_ensemble_model)
        self.btn_save_ensemble.setMinimumHeight(40)
        self.btn_save_ensemble.setEnabled(False)
        ensemble_io_layout.addWidget(self.btn_save_ensemble)
        self.btn_load_ensemble = QPushButton('📂 导入集成模型')
        self.btn_load_ensemble.clicked.connect(self.load_ensemble_model)
        self.btn_load_ensemble.setMinimumHeight(40)
        ensemble_io_layout.addWidget(self.btn_load_ensemble)
        control_panel.addLayout(ensemble_io_layout)
        control_panel.addStretch()

        result_layout = QVBoxLayout()
        self.ensemble_result = QTextEdit()
        self.ensemble_result.setReadOnly(True)
        self.ensemble_result.setFont(QFont("Courier New", 9))
        result_layout.addWidget(self.ensemble_result)

        main_layout.addLayout(control_panel, 1)
        main_layout.addLayout(result_layout, 2)
        self.ensemble_tab.setLayout(main_layout)

    def init_viz_tab(self):
        layout = QVBoxLayout()
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel('图表类型:'))
        self.viz_type_combo = QComboBox()
        self.viz_type_combo.addItems([
            '直方图 (Histogram)',
            '散点图+拟合线 (Scatter+Fit)',
            '箱线图 (Boxplot)',
            '热力图 (Heatmap)',
            '预测vs实际 (Prediction vs Actual)',
            '多变量对比 (Multi-Variable)'
        ])
        self.viz_type_combo.currentIndexChanged.connect(self.update_viz_controls)
        control_layout.addWidget(self.viz_type_combo)

        self.viz_x_label = QLabel('X轴:')
        control_layout.addWidget(self.viz_x_label)
        from widgets import CheckableComboBox
        self.viz_x_combo = CheckableComboBox()
        self.viz_x_combo.setMinimumWidth(200)
        control_layout.addWidget(self.viz_x_combo)

        self.viz_y_label = QLabel('Y轴:')
        control_layout.addWidget(self.viz_y_label)
        self.viz_y_combo = CheckableComboBox()
        self.viz_y_combo.setMinimumWidth(200)
        control_layout.addWidget(self.viz_y_combo)

        self.btn_plot = QPushButton('📊 生成图表')
        self.btn_plot.clicked.connect(self.generate_plot)
        control_layout.addWidget(self.btn_plot)
        control_layout.addStretch()
        layout.addLayout(control_layout)

        self.figure = Figure(figsize=(14, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.viz_tab.setLayout(layout)

    def init_ai_tab(self):
        layout = QVBoxLayout()
        transformers_available = _check_transformers()

        config_group = QGroupBox("🔧 AI配置")
        config_layout = QVBoxLayout()

        model_type_layout = QHBoxLayout()
        self.ai_model_type_group = QButtonGroup()
        self.radio_transformers = QRadioButton("本地模型 (Transformers)")
        self.radio_api = QRadioButton("API调用 (推荐)")
        self.radio_api.setChecked(True)
        self.ai_model_type_group.addButton(self.radio_transformers, 1)
        self.ai_model_type_group.addButton(self.radio_api, 2)
        model_type_layout.addWidget(self.radio_transformers)
        model_type_layout.addWidget(self.radio_api)
        if not transformers_available:
            self.radio_transformers.setEnabled(False)
            self.radio_transformers.setToolTip("需要安装: pip install transformers torch")
        model_type_layout.addStretch()
        config_layout.addLayout(model_type_layout)

        transformers_layout = QHBoxLayout()
        transformers_layout.addWidget(QLabel('模型路径/名称:'))
        self.ai_transformers_path = QLineEdit()
        self.ai_transformers_path.setPlaceholderText(
            'HuggingFace ID 或本地路径，例如: Qwen/Qwen3-0.6B  |  Qwen/Qwen3-1.7B')
        transformers_layout.addWidget(self.ai_transformers_path)
        self.btn_load_transformers = QPushButton('加载模型')
        self.btn_load_transformers.clicked.connect(self.load_transformers_model)
        transformers_layout.addWidget(self.btn_load_transformers)
        config_layout.addLayout(transformers_layout)

        hint_label = QLabel(
            '💡 本地模型提示: 首次加载会自动从 HuggingFace 下载。'
            'Qwen3-0.6B (~1.2GB) 适合CPU运行；Qwen3-1.7B 效果更好但较慢。'
        )
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet("color: #666; font-size: 11px; padding: 2px 0;")
        config_layout.addWidget(hint_label)

        api_layout = QHBoxLayout()
        api_layout.addWidget(QLabel('API 地址:'))
        self.ai_api_url = QLineEdit()
        self.ai_api_url.setText('https://api.siliconflow.cn/v1/chat/completions')
        self.ai_api_url.setPlaceholderText('OpenAI 兼容接口地址')
        api_layout.addWidget(self.ai_api_url)
        config_layout.addLayout(api_layout)

        api_key_layout = QHBoxLayout()
        api_key_layout.addWidget(QLabel('API Key:'))
        self.ai_api_key = QLineEdit()
        self.ai_api_key.setPlaceholderText('在 siliconflow.cn 免费注册后获取 API Key')
        self.ai_api_key.setEchoMode(QLineEdit.Password)
        api_key_layout.addWidget(self.ai_api_key)
        config_layout.addLayout(api_key_layout)

        model_name_layout = QHBoxLayout()
        model_name_layout.addWidget(QLabel('模型名称:'))
        self.ai_model_name = QLineEdit()
        self.ai_model_name.setText('Qwen/Qwen3-8B')
        self.ai_model_name.setPlaceholderText('模型名称')
        model_name_layout.addWidget(self.ai_model_name)

        self.ai_model_preset = QComboBox()
        self.ai_model_preset.setMinimumWidth(230)
        self.ai_model_preset.addItems([
            '-- 快速选择模型 --',
            'Qwen/Qwen3-8B',
            'Qwen/Qwen3-14B',
            'Qwen/Qwen2.5-7B-Instruct',
            'deepseek-ai/DeepSeek-V3',
            'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
            'THUDM/glm-4-9b-chat',
        ])
        self.ai_model_preset.currentIndexChanged.connect(self._on_model_preset_changed)
        model_name_layout.addWidget(self.ai_model_preset)
        config_layout.addLayout(model_name_layout)

        api_hint = QLabel(
            '💡 API提示: 使用硅基流动 (SiliconFlow) 平台，注册即送免费额度，支持 Qwen3、DeepSeek 等主流模型。'
        )
        api_hint.setWordWrap(True)
        api_hint.setStyleSheet("color: #666; font-size: 11px; padding: 2px 0;")
        config_layout.addWidget(api_hint)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        chat_group = QGroupBox("💬 对话 (模型已了解当前数据和模型分析结果，可直接提问)")
        chat_layout = QVBoxLayout()
        self.ai_chat_history = QTextBrowser()
        self.ai_chat_history.setMinimumHeight(300)
        chat_layout.addWidget(self.ai_chat_history)

        input_layout = QHBoxLayout()
        self.ai_input = QLineEdit()
        self.ai_input.setPlaceholderText(
            '例如: 模型准确率如何？有什么改进建议？  /  数据中哪些特征最重要？')
        self.ai_input.returnPressed.connect(self.send_ai_message)
        input_layout.addWidget(self.ai_input)
        self.btn_send = QPushButton('发送')
        self.btn_send.clicked.connect(self.send_ai_message)
        self.btn_send.setMinimumHeight(40)
        input_layout.addWidget(self.btn_send)
        self.btn_clear_chat = QPushButton('清空')
        self.btn_clear_chat.clicked.connect(self.clear_ai_chat)
        self.btn_clear_chat.setMinimumHeight(40)
        input_layout.addWidget(self.btn_clear_chat)
        chat_layout.addLayout(input_layout)
        chat_group.setLayout(chat_layout)
        layout.addWidget(chat_group)
        self.ai_tab.setLayout(layout)

    def _on_model_preset_changed(self, index):
        if index > 0:
            self.ai_model_name.setText(self.ai_model_preset.currentText())
            self.ai_model_preset.setCurrentIndex(0)
