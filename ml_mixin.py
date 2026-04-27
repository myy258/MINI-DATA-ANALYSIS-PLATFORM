import numpy as np
import pandas as pd
from datetime import datetime
from PyQt5.QtWidgets import (QMessageBox, QFileDialog, QCheckBox, QWidget,
                             QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, mean_squared_error, r2_score,
                             confusion_matrix, silhouette_score, classification_report)
import joblib

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class MLMixin:
    def select_all_features(self):
        for i in range(self.feature_checkboxes_layout.count()):
            cb = self.feature_checkboxes_layout.itemAt(i).widget()
            if cb:
                cb.setChecked(True)

    def select_none_features(self):
        for i in range(self.feature_checkboxes_layout.count()):
            cb = self.feature_checkboxes_layout.itemAt(i).widget()
            if cb:
                cb.setChecked(False)

    def update_ml_models(self):
        task = self.ml_task_combo.currentText()
        self.ml_model_combo.clear()
        if '分类' in task:
            models = ['逻辑回归 (Logistic Regression)', '决策树 (Decision Tree)', '随机森林 (Random Forest)']
            if LIGHTGBM_AVAILABLE:
                models.append('LightGBM 分类器')
            self.ml_model_combo.addItems(models)
            self.target_group.show()
            self.cluster_group.hide()
        elif '回归' in task:
            models = ['线性回归 (Linear Regression)', '随机森林回归 (Random Forest Regressor)']
            if LIGHTGBM_AVAILABLE:
                models.append('LightGBM 回归器')
            self.ml_model_combo.addItems(models)
            self.target_group.show()
            self.cluster_group.hide()
        else:
            self.ml_model_combo.addItems(['K-Means 聚类'])
            self.target_group.hide()
            self.cluster_group.show()
        self.update_hyperparameters()

    def get_standard_model_params(self, model_name):
        params = {}
        if '随机森林' in model_name:
            params = {
                'n_estimators': {'type': 'spin', 'min': 10, 'max': 5000, 'default': 100, 'label': 'n_estimators (树的数量)'},
                'max_depth':    {'type': 'spin', 'min': 1,  'max': 50,   'default': 10,  'label': 'max_depth (最大深度)'},
                'min_samples_split': {'type': 'spin', 'min': 2, 'max': 20, 'default': 2, 'label': 'min_samples_split'},
            }
        elif '决策树' in model_name:
            params = {
                'max_depth':         {'type': 'spin', 'min': 1, 'max': 50, 'default': 10, 'label': 'max_depth (最大深度)'},
                'min_samples_split': {'type': 'spin', 'min': 2, 'max': 20, 'default': 2,  'label': 'min_samples_split'},
            }
        elif 'LightGBM' in model_name:
            params = {
                'n_estimators':  {'type': 'spin',   'min': 10,    'max': 5000, 'default': 100,  'label': 'n_estimators (树的数量)'},
                'learning_rate': {'type': 'double', 'min': 0.001, 'max': 1.0,  'default': 0.1,  'step': 0.01, 'label': 'learning_rate (学习率)'},
                'max_depth':     {'type': 'spin',   'min': -1,    'max': 50,   'default': -1,   'label': 'max_depth (最大深度)'},
                'num_leaves':    {'type': 'spin',   'min': 2,     'max': 1000, 'default': 31,   'label': 'num_leaves (叶子数)'},
            }
        elif '逻辑回归' in model_name:
            params = {
                'C':        {'type': 'double', 'min': 0.001, 'max': 100.0, 'default': 1.0,  'step': 0.1, 'label': 'C (正则化强度)'},
                'max_iter': {'type': 'spin',   'min': 100,   'max': 10000, 'default': 1000, 'label': 'max_iter (最大迭代次数)'},
            }
        return params

    def update_hyperparameters(self):
        for i in reversed(range(self.param_layout.count())):
            widget = self.param_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self.param_widgets = {}
        model_name = self.ml_model_combo.currentText()
        params_config = self.get_standard_model_params(model_name)
        if params_config:
            for param_name, config in params_config.items():
                self.add_param_control(param_name, config['label'], config['type'],
                                       config['min'], config['max'], config['default'],
                                       config.get('step'))
        else:
            from PyQt5.QtWidgets import QLabel
            self.param_layout.addWidget(QLabel('该模型无可调超参数'))

    def add_param_control(self, param_name, label, control_type, min_val, max_val, default_val, step=None):
        container = QWidget()
        h_layout = QHBoxLayout()
        h_layout.setContentsMargins(0, 0, 0, 0)
        checkbox = QCheckBox()
        checkbox.setChecked(False)
        h_layout.addWidget(checkbox)
        param_label = QLabel(label + ':')
        param_label.setMinimumWidth(180)
        h_layout.addWidget(param_label)
        if control_type == 'spin':
            control = QSpinBox()
            control.setMinimum(min_val)
            control.setMaximum(max_val)
            control.setValue(default_val)
        elif control_type == 'double':
            control = QDoubleSpinBox()
            control.setMinimum(min_val)
            control.setMaximum(max_val)
            control.setValue(default_val)
            if step:
                control.setSingleStep(step)
        control.setEnabled(False)
        h_layout.addWidget(control)
        h_layout.addStretch()
        checkbox.stateChanged.connect(lambda state: control.setEnabled(state == Qt.Checked))
        container.setLayout(h_layout)
        self.param_layout.addWidget(container)
        self.param_widgets[param_name] = {'checkbox': checkbox, 'control': control}

    def get_model_params(self, model_name):
        params = {'random_state': 42}
        for param_name, widgets in self.param_widgets.items():
            if widgets['checkbox'].isChecked():
                params[param_name] = widgets['control'].value()
        return params

    def train_model(self):
        if self.df is None:
            QMessageBox.warning(self, '警告', '请先加载数据')
            return
        selected_features = []
        for i in range(self.feature_checkboxes_layout.count()):
            cb = self.feature_checkboxes_layout.itemAt(i).widget()
            if cb and cb.isChecked():
                selected_features.append(cb.text())
        if len(selected_features) == 0:
            QMessageBox.warning(self, '警告', '请至少选择一个特征列')
            return
        task = self.ml_task_combo.currentText()
        if '聚类' in task:
            self.perform_clustering(selected_features)
        else:
            target = self.ml_target_combo.currentText()
            if target in selected_features:
                selected_features.remove(target)
            if '分类' in task:
                self.perform_classification(selected_features, target)
            else:
                self.perform_regression(selected_features, target)

    def perform_classification(self, features, target):
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
            X_test_scaled = scaler.transform(X_test)
            model_name = self.ml_model_combo.currentText()
            params = self.get_model_params(model_name)
            if '逻辑回归' in model_name:
                model = LogisticRegression(**params)
            elif '決策树' in model_name or '决策树' in model_name:
                model = DecisionTreeClassifier(**params)
            elif '随机森林' in model_name:
                model = RandomForestClassifier(**params)
            elif 'LightGBM' in model_name:
                model = lgb.LGBMClassifier(**params)
            model.fit(X_train_scaled, y_train)
            self.trained_model = model
            self.trained_scaler = scaler

            y_train_pred = model.predict(X_train_scaled)
            y_test_pred  = model.predict(X_test_scaled)
            # Store both train and test results for visualization
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
            cm_test = confusion_matrix(y_test, y_test_pred)

            result  = "=" * 90 + f"\n分类模型训练结果 - {model_name}\n" + "=" * 90 + "\n\n"
            result += f"特征列: {', '.join(features)}\n目标列: {target}\n"
            result += f"训练集大小: {len(X_train)} | 测试集大小: {len(X_test)}\n类别数量: {n_classes}\n\n"
            result += "【模型超参数】\n" + "-" * 90 + "\n"
            result += ("\n".join(f"{k}: {v}" for k, v in params.items()) if len(params) > 1 else "使用默认参数") + "\n\n"
            result += "【模型性能对比 - 训练集 vs 测试集】\n" + "-" * 90 + "\n"
            result += f"{'指标':<25} {'训练集':>15} {'测试集':>15} {'差异':>15} {'过拟合程度':>15}\n" + "-" * 90 + "\n"
            acc_diff    = train_accuracy - test_accuracy
            acc_overfit = "高" if acc_diff > 0.15 else ("中" if acc_diff > 0.08 else "低")
            result += f"{'准确率 (Accuracy)':<25} {train_accuracy:>15.4f} {test_accuracy:>15.4f} {acc_diff:>+15.4f} {acc_overfit:>15}\n"
            result += f"{'精确率 (Precision)':<25} {train_precision:>15.4f} {test_precision:>15.4f} {train_precision-test_precision:>+15.4f}\n"
            result += f"{'召回率 (Recall)':<25} {train_recall:>15.4f} {test_recall:>15.4f} {train_recall-test_recall:>+15.4f}\n"
            result += f"{'F1分数 (F1-Score)':<25} {train_f1:>15.4f} {test_f1:>15.4f} {train_f1-test_f1:>+15.4f}\n\n"
            result += "【模型诊断】\n" + "-" * 90 + "\n"
            if acc_diff > 0.15:
                result += ("⚠️  严重过拟合: 训练集准确率显著高于测试集 (差异 > 15%)\n"
                           "   📌 建议措施:\n      1. 增加正则化强度\n      2. 使用交叉验证\n"
                           "      3. 增加训练数据量\n      4. 特征降维或特征选择\n\n")
            elif acc_diff > 0.08:
                result += "⚠️  轻度过拟合: 训练集表现略优于测试集 (差异 8%-15%)\n   📌 建议: 适当调整正则化参数\n\n"
            elif test_accuracy < 0.6:
                result += ("⚠️  欠拟合: 整体准确率较低 (< 60%)\n   📌 建议措施:\n"
                           "      1. 增加模型复杂度\n      2. 添加更多特征\n"
                           "      3. 调整超参数\n      4. 检查数据质量\n\n")
            else:
                result += "✓ 模型泛化良好: 训练集与测试集性能均衡\n\n"
            if n_classes <= 10:
                result += "【测试集 - 各类别详细指标】\n" + "-" * 90 + "\n"
                result += classification_report(y_test, y_test_pred, zero_division=0) + "\n"
            result += "【混淆矩阵 - 测试集】\n" + "-" * 90 + "\n" + str(cm_test) + "\n\n"
            if hasattr(model, 'feature_importances_'):
                result += "【特征重要性 Top 10】\n" + "-" * 90 + "\n"
                importance = pd.DataFrame({'特征': features, '重要性': model.feature_importances_}
                                         ).sort_values('重要性', ascending=False).head(10)
                result += importance.to_string(index=False) + "\n"
            result += "\n💡 可视化提示: 在「数据可视化」选项卡查看预测效果\n💾 导出提示: 点击「导出模型」保存训练好的模型\n"
            self.ml_result.setText(result)
            self.tabs.setCurrentWidget(self.ml_tab)
            self.btn_save_model.setEnabled(True)
            self.btn_reset.setEnabled(True)
        except Exception as e:
            raise Exception(f"分类任务执行失败: {str(e)}")

    def perform_regression(self, features, target):
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
            model_name = self.ml_model_combo.currentText()
            params     = self.get_model_params(model_name)
            if '线性回归' in model_name:
                model = LinearRegression()
            elif '随机森林' in model_name:
                model = RandomForestRegressor(**params)
            elif 'LightGBM' in model_name:
                model = lgb.LGBMRegressor(**params)
            model.fit(X_train_scaled, y_train)
            self.trained_model  = model
            self.trained_scaler = scaler

            y_train_pred = model.predict(X_train_scaled)
            y_test_pred  = model.predict(X_test_scaled)
            # Store both train and test results for visualization
            self.last_train_predictions = y_train_pred
            self.last_train_actual      = y_train.values if hasattr(y_train, 'values') else y_train
            self.last_predictions       = y_test_pred
            self.last_actual            = y_test.values  if hasattr(y_test,  'values') else y_test

            train_r2   = r2_score(y_train, y_train_pred)
            train_mse  = mean_squared_error(y_train, y_train_pred)
            train_rmse = np.sqrt(train_mse)
            train_mae  = np.mean(np.abs(y_train - y_train_pred))
            train_mape = np.mean(np.abs((y_train - y_train_pred) / (y_train + 1e-10))) * 100
            test_r2    = r2_score(y_test, y_test_pred)
            test_mse   = mean_squared_error(y_test, y_test_pred)
            test_rmse  = np.sqrt(test_mse)
            test_mae   = np.mean(np.abs(y_test - y_test_pred))
            test_mape  = np.mean(np.abs((y_test - y_test_pred) / (y_test + 1e-10))) * 100

            result  = "=" * 95 + f"\n回归模型训练结果 - {model_name}\n" + "=" * 95 + "\n\n"
            result += f"特征列: {', '.join(features)}\n目标列: {target}\n"
            result += f"训练集大小: {len(X_train)} | 测试集大小: {len(X_test)}\n\n"
            if len(params) > 1:
                result += "【模型超参数】\n" + "-" * 95 + "\n"
                result += "\n".join(f"{k}: {v}" for k, v in params.items()) + "\n\n"
            result += "【模型性能对比 - 训练集 vs 测试集】\n" + "-" * 95 + "\n"
            result += f"{'指标':<30} {'训练集':>15} {'测试集':>15} {'差异':>15} {'过拟合程度':>15}\n" + "-" * 95 + "\n"
            r2_diff    = train_r2 - test_r2
            r2_overfit = "高" if r2_diff > 0.2 else ("中" if r2_diff > 0.1 else "低")
            result += f"{'R² 决定系数':<30} {train_r2:>15.4f} {test_r2:>15.4f} {r2_diff:>+15.4f} {r2_overfit:>15}\n"
            result += f"{'均方误差 (MSE)':<30} {train_mse:>15.4f} {test_mse:>15.4f} {train_mse-test_mse:>+15.4f}\n"
            result += f"{'均方根误差 (RMSE)':<30} {train_rmse:>15.4f} {test_rmse:>15.4f} {train_rmse-test_rmse:>+15.4f}\n"
            result += f"{'平均绝对误差 (MAE)':<30} {train_mae:>15.4f} {test_mae:>15.4f} {train_mae-test_mae:>+15.4f}\n"
            result += f"{'平均绝对百分比误差 (MAPE)':<30} {train_mape:>14.2f}% {test_mape:>14.2f}% {train_mape-test_mape:>+14.2f}%\n\n"
            result += "【模型诊断】\n" + "-" * 95 + "\n"
            if r2_diff > 0.2:
                result += ("⚠️  严重过拟合: 训练集R²显著高于测试集 (差异 > 0.2)\n"
                           "   📌 建议措施:\n      1. 减少模型复杂度\n      2. 增加正则化\n"
                           "      3. 使用交叉验证\n      4. 收集更多训练数据\n\n")
            elif r2_diff > 0.1:
                result += "⚠️  轻度过拟合 (差异 0.1-0.2)\n   📌 建议: 适当简化模型或增加正则化\n\n"
            elif test_r2 < 0.5:
                result += ("⚠️  欠拟合: 测试集R²较低 (< 0.5)\n   📌 建议措施:\n"
                           "      1. 增加模型复杂度\n      2. 特征工程\n"
                           "      3. 检查数据质量\n      4. 尝试其他算法\n\n")
            elif test_r2 > 0.8:
                result += "✓ 模型表现优秀: R² > 0.8，拟合效果良好\n\n"
            else:
                result += "✓ 模型泛化良好: 训练集与测试集性能均衡\n\n"
            y_test_values = y_test.values if hasattr(y_test, 'values') else y_test
            comparison = pd.DataFrame({
                '实际值': y_test_values[:10], '预测值': y_test_pred[:10],
                '误差': y_test_values[:10] - y_test_pred[:10],
                '相对误差%': ((y_test_values[:10] - y_test_pred[:10]) / (y_test_values[:10] + 1e-10) * 100)
            })
            result += "【预测样例对比 - 测试集前10个】\n" + "-" * 95 + "\n" + comparison.to_string() + "\n\n"
            residuals = y_test_values - y_test_pred
            result += ("【残差统计分析】\n" + "-" * 95 + "\n"
                       f"残差均值:   {np.mean(residuals):>12.4f}\n残差标准差: {np.std(residuals):>12.4f}\n"
                       f"残差最大值: {np.max(residuals):>12.4f}\n残差最小值: {np.min(residuals):>12.4f}\n\n")
            if hasattr(model, 'feature_importances_'):
                result += "【特征重要性 Top 10】\n" + "-" * 95 + "\n"
                importance = pd.DataFrame({'特征': features, '重要性': model.feature_importances_}
                                         ).sort_values('重要性', ascending=False).head(10)
                result += importance.to_string(index=False) + "\n"
            elif hasattr(model, 'coef_'):
                result += "【回归系数】\n" + "-" * 95 + "\n"
                coefs = pd.DataFrame({'特征': features, '系数': model.coef_}).sort_values('系数', key=abs, ascending=False)
                result += coefs.to_string(index=False) + f"\n\n截距 (Intercept): {model.intercept_:.4f}\n"
            result += "\n💡 可视化提示: 在「数据可视化」选项卡查看预测vs实际散点图\n💾 导出提示: 点击「导出模型」保存训练好的模型\n"
            self.ml_result.setText(result)
            self.tabs.setCurrentWidget(self.ml_tab)
            self.btn_save_model.setEnabled(True)
            self.btn_reset.setEnabled(True)
        except Exception as e:
            raise Exception(f"回归任务执行失败: {str(e)}")

    def perform_clustering(self, features):
        try:
            X = self.df[features].copy()
            X = X.fillna(X.mean(numeric_only=True))
            le = LabelEncoder()
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = le.fit_transform(X[col].astype(str))
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            n_clusters = self.cluster_n.value()
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = model.fit_predict(X_scaled)
            self.trained_model  = model
            self.trained_scaler = scaler
            silhouette = silhouette_score(X_scaled, clusters)
            inertia    = model.inertia_
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            result  = "=" * 80 + f"\nK-Means 聚类结果\n" + "=" * 80 + "\n\n"
            result += f"特征列: {', '.join(features)}\n聚类数量: {n_clusters}\n数据点数量: {len(X)}\n\n"
            result += "【聚类性能】\n" + "-" * 80 + "\n"
            result += f"轮廓系数 (Silhouette Score): {silhouette:.4f}\n  (范围: -1 到 1, 越接近1表示聚类效果越好)\n"
            result += f"簇内平方和 (Inertia):        {inertia:.2f}\n\n"
            result += "【各簇样本数量】\n" + "-" * 80 + "\n"
            for cluster_id in range(n_clusters):
                count = cluster_counts.get(cluster_id, 0)
                result += f"簇 {cluster_id}: {count} 个样本 ({count/len(X)*100:.2f}%)\n"
            result += "\n【簇中心坐标】\n" + "-" * 80 + "\n"
            centers_df = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=features)
            result += centers_df.to_string() + "\n"
            result += f"\n✓ 聚类标签已添加到数据中 (列名: 'Cluster')\n  您可以在「数据查看」选项卡中查看\n"
            result += "\n💡 提示: 点击「导出模型」按钮可保存训练好的模型\n"
            self.df['Cluster'] = clusters
            self.ml_result.setText(result)
            self.tabs.setCurrentWidget(self.ml_tab)
            self.btn_save_model.setEnabled(True)
            self.btn_reset.setEnabled(True)
            self.display_data(self.df)
            self.update_ml_feature_list()
        except Exception as e:
            raise Exception(f"聚类任务执行失败: {str(e)}")

    def reset_ml_results(self):
        from PyQt5.QtWidgets import QMessageBox
        reply = QMessageBox.question(self, '确认重置', '确定要清除所有训练结果吗？',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.trained_model = self.trained_scaler = None
            self.last_predictions = self.last_actual = None
            self.last_train_predictions = self.last_train_actual = None
            self.ml_result.clear()
            self.btn_save_model.setEnabled(False)
            self.btn_reset.setEnabled(False)
            QMessageBox.information(self, '成功', '训练结果已清除')

    def save_model(self):
        if self.trained_model is None:
            QMessageBox.warning(self, '警告', '没有可导出的模型，请先训练模型')
            return
        try:
            model_name  = self.ml_model_combo.currentText()
            model_type  = ("LogisticRegression" if '逻辑回归' in model_name else
                           "LinearRegression"   if '线性回归' in model_name else
                           "DecisionTree"       if '决策树'  in model_name else
                           "RandomForestRegressor" if '随机森林回归' in model_name else
                           "RandomForestClassifier" if '随机森林' in model_name else
                           "KMeans"             if 'K-Means' in model_name else
                           "LGBMClassifier"     if 'LightGBM' in model_name and ('分类' in model_name or '分类器' in model_name) else
                           "LGBMRegressor")
            date_str     = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存模型", f"{model_type}_{date_str}.pkl",
                "Model Files (*.pkl);;All Files (*)")
            if file_path:
                joblib.dump({'model': self.trained_model, 'scaler': self.trained_scaler,
                             'model_name': model_name,
                             'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, file_path)
                QMessageBox.information(self, '成功', f'模型已成功导出到:\n{file_path}')
                self.statusBar().showMessage(f'✓ 模型已导出: {file_path}')
        except Exception as e:
            QMessageBox.critical(self, '错误', f'模型导出失败:\n{str(e)}')

    def load_model(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择模型文件", "", "Model Files (*.pkl);;All Files (*)")
            if file_path:
                model_data = joblib.load(file_path)
                self.trained_model  = model_data['model']
                self.trained_scaler = model_data['scaler']
                model_name  = model_data.get('model_name', 'Unknown')
                train_date  = model_data.get('date', 'Unknown')
                self.btn_save_model.setEnabled(True)
                self.btn_reset.setEnabled(True)
                result = ("=" * 80 + "\n模型导入成功\n" + "=" * 80 + "\n\n"
                          f"模型类型: {model_name}\n训练日期: {train_date}\n文件路径: {file_path}\n\n"
                          "✓ 模型已加载，可以使用该模型进行预测\n"
                          "\n💡 提示: 使用导入的模型前，请确保数据格式与训练时一致\n")
                self.ml_result.setText(result)
                self.tabs.setCurrentWidget(self.ml_tab)
                QMessageBox.information(self, '成功', f'模型已成功导入\n\n模型类型: {model_name}\n训练日期: {train_date}')
                self.statusBar().showMessage(f'✓ 模型已导入: {model_name}')
        except Exception as e:
            QMessageBox.critical(self, '错误', f'模型导入失败:\n{str(e)}')