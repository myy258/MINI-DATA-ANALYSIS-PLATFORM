import numpy as np
import pandas as pd
from datetime import datetime
from PyQt5.QtWidgets import (QMessageBox, QFileDialog, QDialog, QDialogButtonBox,
                             QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
                             QDoubleSpinBox, QGroupBox, QScrollArea, QWidget,
                             QCheckBox, QSpinBox)
from PyQt5.QtCore import Qt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               VotingClassifier, VotingRegressor)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, mean_squared_error, r2_score,
                             confusion_matrix)
import joblib

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class EnsembleMixin:
    def select_all_ensemble_features(self):
        for i in range(self.ensemble_feature_checkboxes_layout.count()):
            cb = self.ensemble_feature_checkboxes_layout.itemAt(i).widget()
            if cb:
                cb.setChecked(True)

    def select_none_ensemble_features(self):
        for i in range(self.ensemble_feature_checkboxes_layout.count()):
            cb = self.ensemble_feature_checkboxes_layout.itemAt(i).widget()
            if cb:
                cb.setChecked(False)

    def add_ensemble_model(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('添加模型到集成')
        dialog.setMinimumWidth(500)
        layout = QVBoxLayout()

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel('选择模型:'))
        model_combo = QComboBox()
        task = self.ensemble_task_combo.currentText()
        if '分类' in task:
            models = ['逻辑回归', '决策树', '随机森林']
            if LIGHTGBM_AVAILABLE:
                models.append('LightGBM')
        else:
            models = ['线性回归', '随机森林回归']
            if LIGHTGBM_AVAILABLE:
                models.append('LightGBM回归')
        model_combo.addItems(models)
        model_layout.addWidget(model_combo)
        layout.addLayout(model_layout)

        weight_layout = QHBoxLayout()
        weight_layout.addWidget(QLabel('模型权重:'))
        weight_spin = QDoubleSpinBox()
        weight_spin.setMinimum(0.1)
        weight_spin.setMaximum(10.0)
        weight_spin.setValue(1.0)
        weight_spin.setSingleStep(0.1)
        weight_layout.addWidget(weight_spin)
        layout.addLayout(weight_layout)

        param_group = QGroupBox("模型参数 (可选)")
        param_scroll = QScrollArea()
        param_scroll.setWidgetResizable(True)
        param_scroll.setMaximumHeight(200)
        param_widget = QWidget()
        param_layout = QVBoxLayout()
        param_widget.setLayout(param_layout)
        param_scroll.setWidget(param_widget)
        param_group_layout = QVBoxLayout()
        param_group_layout.addWidget(param_scroll)
        param_group.setLayout(param_group_layout)
        layout.addWidget(param_group)

        param_widgets = {}
        def update_params():
            for i in reversed(range(param_layout.count())):
                w = param_layout.itemAt(i).widget()
                if w:
                    w.setParent(None)
            param_widgets.clear()
            params_config = self.get_standard_model_params(model_combo.currentText())
            for param_name, config in params_config.items():
                container = QWidget()
                h_layout = QHBoxLayout()
                h_layout.setContentsMargins(0, 0, 0, 0)
                checkbox = QCheckBox()
                h_layout.addWidget(checkbox)
                lbl = QLabel(config['label'] + ':')
                lbl.setMinimumWidth(180)
                h_layout.addWidget(lbl)
                if config['type'] == 'spin':
                    control = QSpinBox()
                    control.setMinimum(config['min'])
                    control.setMaximum(config['max'])
                    control.setValue(config['default'])
                else:
                    control = QDoubleSpinBox()
                    control.setMinimum(config['min'])
                    control.setMaximum(config['max'])
                    control.setValue(config['default'])
                    if 'step' in config:
                        control.setSingleStep(config['step'])
                control.setEnabled(False)
                h_layout.addWidget(control)
                h_layout.addStretch()
                checkbox.stateChanged.connect(lambda state, c=control: c.setEnabled(state == Qt.Checked))
                container.setLayout(h_layout)
                param_layout.addWidget(container)
                param_widgets[param_name] = {'checkbox': checkbox, 'control': control}

        model_combo.currentIndexChanged.connect(update_params)
        update_params()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.setLayout(layout)

        if dialog.exec_() == QDialog.Accepted:
            model_name = model_combo.currentText()
            weight     = weight_spin.value()
            params = {pn: pw['control'].value()
                      for pn, pw in param_widgets.items() if pw['checkbox'].isChecked()}
            self.ensemble_models.append({'name': model_name, 'weight': weight, 'params': params})
            self.ensemble_models_list.addItem(f"{model_name} (权重: {weight})")
            QMessageBox.information(self, '成功', f'已添加模型: {model_name}')

    def remove_ensemble_model(self):
        current_row = self.ensemble_models_list.currentRow()
        if current_row >= 0:
            self.ensemble_models_list.takeItem(current_row)
            del self.ensemble_models[current_row]
            QMessageBox.information(self, '成功', '已移除选中模型')
        else:
            QMessageBox.warning(self, '警告', '请先选择要移除的模型')

    def train_ensemble_model(self):
        if self.df is None:
            QMessageBox.warning(self, '警告', '请先加载数据')
            return
        if len(self.ensemble_models) == 0:
            QMessageBox.warning(self, '警告', '请先添加至少一个模型到集成')
            return
        try:
            selected_features = []
            for i in range(self.ensemble_feature_checkboxes_layout.count()):
                cb = self.ensemble_feature_checkboxes_layout.itemAt(i).widget()
                if cb and cb.isChecked():
                    selected_features.append(cb.text())
            if len(selected_features) == 0:
                QMessageBox.warning(self, '警告', '请至少选择一个特征列')
                return
            target = self.ensemble_target_combo.currentText()
            if target in selected_features:
                selected_features.remove(target)
            task = self.ensemble_task_combo.currentText()
            if '分类' in task:
                self.perform_ensemble_classification(selected_features, target)
            else:
                self.perform_ensemble_regression(selected_features, target)
        except Exception as e:
            QMessageBox.critical(self, '错误', f'集成模型训练失败:\n{str(e)}')

    def perform_ensemble_classification(self, features, target):
        try:
            X = self.df[features].copy()
            y = self.df[target].copy()
            X = X.fillna(X.mean(numeric_only=True))
            y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0)
            le = LabelEncoder()
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = le.fit_transform(X[col].astype(str))
            if y.dtype == 'object':
                y = le.fit_transform(y.astype(str))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled  = scaler.transform(X_test)
            estimators = []
            weights    = []
            for model_info in self.ensemble_models:
                mn     = model_info['name']
                params = {**model_info['params'], 'random_state': 42}
                if '逻辑回归' in mn:
                    m = LogisticRegression(**params)
                elif '决策树' in mn:
                    m = DecisionTreeClassifier(**params)
                elif '随机森林' in mn:
                    m = RandomForestClassifier(**params)
                elif 'LightGBM' in mn:
                    m = lgb.LGBMClassifier(**params)
                estimators.append((mn, m))
                weights.append(model_info['weight'])
            voting_clf = VotingClassifier(estimators=estimators, voting='soft', weights=weights)
            voting_clf.fit(X_train_scaled, y_train)
            self.trained_ensemble_model  = voting_clf
            self.trained_ensemble_scaler = scaler

            y_train_pred = voting_clf.predict(X_train_scaled)
            y_test_pred  = voting_clf.predict(X_test_scaled)
            # Store both sets for visualization
            self.last_train_predictions = y_train_pred
            self.last_train_actual      = y_train
            self.last_predictions       = y_test_pred
            self.last_actual            = y_test

            n_classes  = len(np.unique(y))
            avg_method = 'binary' if n_classes == 2 else 'weighted'
            train_accuracy  = accuracy_score(y_train, y_train_pred)
            train_precision = precision_score(y_train, y_train_pred, average=avg_method, zero_division=0)
            train_recall    = recall_score(y_train, y_train_pred, average=avg_method, zero_division=0)
            train_f1        = f1_score(y_train, y_train_pred, average=avg_method, zero_division=0)
            test_accuracy   = accuracy_score(y_test, y_test_pred)
            test_precision  = precision_score(y_test, y_test_pred, average=avg_method, zero_division=0)
            test_recall     = recall_score(y_test, y_test_pred, average=avg_method, zero_division=0)
            test_f1         = f1_score(y_test, y_test_pred, average=avg_method, zero_division=0)
            cm = confusion_matrix(y_test, y_test_pred)

            result  = "=" * 90 + "\nVoting集成分类模型训练结果\n" + "=" * 90 + "\n\n"
            result += f"特征列: {', '.join(features)}\n目标列: {target}\n"
            result += f"训练集大小: {len(X_train)} | 测试集大小: {len(X_test)}\n\n"
            result += "【集成模型配置】\n" + "-" * 90 + "\n"
            for mi in self.ensemble_models:
                result += f"模型: {mi['name']}, 权重: {mi['weight']}\n"
                if mi['params']:
                    result += f"  参数: {mi['params']}\n"
            result += "\n【模型性能对比 - 训练集 vs 测试集】\n" + "-" * 90 + "\n"
            result += f"{'指标':<25} {'训练集':>15} {'测试集':>15} {'差异':>15}\n" + "-" * 90 + "\n"
            result += f"{'准确率 (Accuracy)':<25} {train_accuracy:>15.4f} {test_accuracy:>15.4f} {train_accuracy-test_accuracy:>+15.4f}\n"
            result += f"{'精确率 (Precision)':<25} {train_precision:>15.4f} {test_precision:>15.4f} {train_precision-test_precision:>+15.4f}\n"
            result += f"{'召回率 (Recall)':<25} {train_recall:>15.4f} {test_recall:>15.4f} {train_recall-test_recall:>+15.4f}\n"
            result += f"{'F1分数 (F1-Score)':<25} {train_f1:>15.4f} {test_f1:>15.4f} {train_f1-test_f1:>+15.4f}\n\n"
            result += "【混淆矩阵 - 测试集】\n" + "-" * 90 + "\n" + str(cm) + "\n\n"
            result += "💡 可视化提示: 在「数据可视化」选项卡查看预测效果\n💾 导出提示: 点击「导出集成模型」保存训练好的模型\n"
            self.ensemble_result.setText(result)
            self.tabs.setCurrentWidget(self.ensemble_tab)
            self.btn_save_ensemble.setEnabled(True)
            self.btn_reset_ensemble.setEnabled(True)
        except Exception as e:
            raise Exception(f"集成分类任务执行失败: {str(e)}")

    def perform_ensemble_regression(self, features, target):
        try:
            X = self.df[features].copy()
            y = self.df[target].copy()
            X = X.fillna(X.mean(numeric_only=True))
            y = y.fillna(y.mean())
            le = LabelEncoder()
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = le.fit_transform(X[col].astype(str))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled  = scaler.transform(X_test)
            estimators = []
            weights    = []
            for model_info in self.ensemble_models:
                mn     = model_info['name']
                params = model_info['params'].copy()
                if mn != '线性回归':
                    params['random_state'] = 42
                if '线性回归' in mn:
                    m = LinearRegression()
                elif '随机森林' in mn:
                    m = RandomForestRegressor(**params)
                elif 'LightGBM' in mn:
                    m = lgb.LGBMRegressor(**params)
                estimators.append((mn, m))
                weights.append(model_info['weight'])
            voting_reg = VotingRegressor(estimators=estimators, weights=weights)
            voting_reg.fit(X_train_scaled, y_train)
            self.trained_ensemble_model  = voting_reg
            self.trained_ensemble_scaler = scaler

            y_train_pred = voting_reg.predict(X_train_scaled)
            y_test_pred  = voting_reg.predict(X_test_scaled)
            # Store both sets for visualization
            self.last_train_predictions = y_train_pred
            self.last_train_actual      = y_train.values if hasattr(y_train, 'values') else y_train
            self.last_predictions       = y_test_pred
            self.last_actual            = y_test.values  if hasattr(y_test,  'values') else y_test

            train_r2   = r2_score(y_train, y_train_pred)
            train_mse  = mean_squared_error(y_train, y_train_pred)
            train_rmse = np.sqrt(train_mse)
            train_mae  = np.mean(np.abs(y_train - y_train_pred))
            test_r2    = r2_score(y_test, y_test_pred)
            test_mse   = mean_squared_error(y_test, y_test_pred)
            test_rmse  = np.sqrt(test_mse)
            test_mae   = np.mean(np.abs(y_test - y_test_pred))

            result  = "=" * 90 + "\nVoting集成回归模型训练结果\n" + "=" * 90 + "\n\n"
            result += f"特征列: {', '.join(features)}\n目标列: {target}\n"
            result += f"训练集大小: {len(X_train)} | 测试集大小: {len(X_test)}\n\n"
            result += "【集成模型配置】\n" + "-" * 90 + "\n"
            for mi in self.ensemble_models:
                result += f"模型: {mi['name']}, 权重: {mi['weight']}\n"
                if mi['params']:
                    result += f"  参数: {mi['params']}\n"
            result += "\n【模型性能对比 - 训练集 vs 测试集】\n" + "-" * 90 + "\n"
            result += f"{'指标':<30} {'训练集':>15} {'测试集':>15} {'差异':>15}\n" + "-" * 90 + "\n"
            result += f"{'R² 决定系数':<30} {train_r2:>15.4f} {test_r2:>15.4f} {train_r2-test_r2:>+15.4f}\n"
            result += f"{'均方误差 (MSE)':<30} {train_mse:>15.4f} {test_mse:>15.4f} {train_mse-test_mse:>+15.4f}\n"
            result += f"{'均方根误差 (RMSE)':<30} {train_rmse:>15.4f} {test_rmse:>15.4f} {train_rmse-test_rmse:>+15.4f}\n"
            result += f"{'平均绝对误差 (MAE)':<30} {train_mae:>15.4f} {test_mae:>15.4f} {train_mae-test_mae:>+15.4f}\n\n"
            result += "💡 可视化提示: 在「数据可视化」选项卡查看预测vs实际散点图\n💾 导出提示: 点击「导出集成模型」保存训练好的模型\n"
            self.ensemble_result.setText(result)
            self.tabs.setCurrentWidget(self.ensemble_tab)
            self.btn_save_ensemble.setEnabled(True)
            self.btn_reset_ensemble.setEnabled(True)
        except Exception as e:
            raise Exception(f"集成回归任务执行失败: {str(e)}")

    def save_ensemble_model(self):
        if self.trained_ensemble_model is None:
            QMessageBox.warning(self, '警告', '没有可导出的集成模型，请先训练模型')
            return
        try:
            task       = self.ensemble_task_combo.currentText()
            model_type = "VotingClassifier" if '分类' in task else "VotingRegressor"
            date_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存集成模型", f"{model_type}_{date_str}.pkl",
                "Model Files (*.pkl);;All Files (*)")
            if file_path:
                joblib.dump({
                    'model': self.trained_ensemble_model,
                    'scaler': self.trained_ensemble_scaler,
                    'ensemble_models': self.ensemble_models,
                    'task': task,
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, file_path)
                QMessageBox.information(self, '成功', f'集成模型已成功导出到:\n{file_path}')
                self.statusBar().showMessage(f'✓ 集成模型已导出: {file_path}')
        except Exception as e:
            QMessageBox.critical(self, '错误', f'集成模型导出失败:\n{str(e)}')

    def load_ensemble_model(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择集成模型文件", "", "Model Files (*.pkl);;All Files (*)")
            if file_path:
                model_data = joblib.load(file_path)
                self.trained_ensemble_model  = model_data['model']
                self.trained_ensemble_scaler = model_data['scaler']
                self.ensemble_models         = model_data.get('ensemble_models', [])
                task       = model_data.get('task', 'Unknown')
                train_date = model_data.get('date', 'Unknown')
                self.ensemble_models_list.clear()
                for mi in self.ensemble_models:
                    self.ensemble_models_list.addItem(f"{mi['name']} (权重: {mi['weight']})")
                self.btn_save_ensemble.setEnabled(True)
                self.btn_reset_ensemble.setEnabled(True)
                result = ("=" * 80 + "\n集成模型导入成功\n" + "=" * 80 + "\n\n"
                          f"任务类型: {task}\n训练日期: {train_date}\n文件路径: {file_path}\n\n"
                          "【包含的模型】\n" + "-" * 80 + "\n")
                for mi in self.ensemble_models:
                    result += f"- {mi['name']} (权重: {mi['weight']})\n"
                result += "\n✓ 集成模型已加载，可以使用该模型进行预测\n"
                self.ensemble_result.setText(result)
                self.tabs.setCurrentWidget(self.ensemble_tab)
                QMessageBox.information(
                    self, '成功',
                    f'集成模型已成功导入\n\n任务类型: {task}\n训练日期: {train_date}\n模型数量: {len(self.ensemble_models)}')
                self.statusBar().showMessage('✓ 集成模型已导入')
        except Exception as e:
            QMessageBox.critical(self, '错误', f'集成模型导入失败:\n{str(e)}')

    def reset_ensemble_results(self):
        reply = QMessageBox.question(self, '确认重置', '确定要清除所有集成模型结果吗？',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.ensemble_models.clear()
            self.ensemble_models_list.clear()
            self.ensemble_result.clear()
            self.trained_ensemble_model  = None
            self.trained_ensemble_scaler = None
            self.last_train_predictions  = None
            self.last_train_actual       = None
            self.btn_reset_ensemble.setEnabled(False)
            self.btn_save_ensemble.setEnabled(False)
            QMessageBox.information(self, '成功', '集成模型结果已清除')